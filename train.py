"""
Fine-tune the VideoMAE -> visual mapper -> GPT-2 pipeline on How2Sign.

Usage:
    python train.py \
        --train_csv /data/how2sign/train.csv \
        --val_csv   /data/how2sign/val.csv \
        --video_root /data/how2sign/clips \
        --output_dir ./checkpoints \
        --epochs 10 --batch_size 4 --lr 3e-5

Notes:
- The VideoMAE encoder is frozen by default (freeze_encoder=True in model.py)
  since How2Sign (~35k clips) is small relative to what's needed to safely
  fine-tune a ViT-scale video encoder end-to-end. Only the visual mapper and
  GPT-2 are trained. Unfreeze the encoder later with a small LR once the
  mapper has converged, if you have the compute budget.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import How2SignDataset, make_collate_fn
from model import SignTranslationModel, build_image_processor


def evaluate(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(pixel_values, input_ids, attention_mask)
            total_loss += out.loss.item()
            n += 1
    model.train()
    return total_loss / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--output_dir", default="./checkpoints")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--num_visual_tokens", type=int, default=16)
    ap.add_argument("--gpt2_name", default="gpt2")
    ap.add_argument("--freeze_encoder", action="store_true", default=True)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    model = SignTranslationModel(
        gpt2_name=args.gpt2_name,
        num_visual_tokens=args.num_visual_tokens,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    image_processor = build_image_processor()
    collate_fn = make_collate_fn(image_processor)

    train_ds = How2SignDataset(args.train_csv, args.video_root, model.tokenizer, num_frames=args.num_frames)
    val_ds = How2SignDataset(args.val_csv, args.video_root, model.tokenizer, num_frames=args.num_frames)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, collate_fn=collate_fn)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable_params) / 1e6:.1f}M")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=max(total_steps, 1))

    best_val = float("inf")
    step = 0
    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        optimizer.zero_grad()
        for i, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            out = model(pixel_values, input_ids, attention_mask)
            loss = out.loss / args.grad_accum
            loss.backward()

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

            if i % args.log_every == 0:
                pbar.set_postfix(loss=out.loss.item())

        val_loss = evaluate(model, val_loader, device)
        print(f"[epoch {epoch}] val_loss={val_loss:.4f}")

        ckpt_path = os.path.join(args.output_dir, f"epoch{epoch}.pt")
        torch.save({"model_state": model.state_dict(), "args": vars(args)}, ckpt_path)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "args": vars(args)},
                       os.path.join(args.output_dir, "best.pt"))
            print(f"  -> new best, saved to {args.output_dir}/best.pt")


if __name__ == "__main__":
    main()
