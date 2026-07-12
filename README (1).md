# ASL -> English streaming translation

Pipeline: **How2Sign** (data) -> **VideoMAE** (frozen video encoder) ->
**learned-query pooler** (compresses patch tokens into K soft visual
tokens) -> **GPT-2** (decoder, soft-prompted with those visual tokens) ->
**sliding-window streaming inference**.

## Why this isn't literally Uni-Sign

Uni-Sign's real architecture pretrains on pose sequences across multiple
sign languages with a gloss-free unified pretraining objective, then
fine-tunes per-language — it needs infrastructure (pose extraction
pipeline, multi-SL pretraining corpus) beyond a single How2Sign fine-tune.
What's implemented here follows **Sign2GPT**'s simpler recipe instead,
since it maps directly onto "How2Sign clips in, English sentences out"
with one dataset and one GPU:

- Frozen pretrained visual encoder (VideoMAE instead of Sign2GPT's DINOv2,
  since VideoMAE is spatio-temporal and doesn't need a separate temporal
  module bolted on).
- A small trainable adapter ("visual mapper") that projects encoder
  features into the decoder's embedding space as a handful of soft
  prompt tokens.
- A frozen-vocab causal LM (GPT-2) decodes English conditioned on that
  prompt.

Only the mapper (and optionally GPT-2's weights) get fine-tuned, which is
realistic on a single T4/A100 with How2Sign's ~35k train clips. If you
later want true Uni-Sign, swap `model.py`'s encoder branch for a pose
estimator (e.g. MediaPipe Holistic) + the pose tokenizer from the Uni-Sign
repo — the GPT-2 decoder half stays the same.

## Files

- `dataset.py` — How2Sign clip/sentence loader (decord-based frame sampling).
- `model.py` — VideoMAE encoder + visual mapper + GPT-2 decoder.
- `train.py` — fine-tuning loop (mapper + GPT-2 trainable, encoder frozen).
- `stream_infer.py` — webcam sliding-window inference loop.

## Setup

```bash
pip install -r requirements.txt
```

Get How2Sign clips + CSVs from https://how2sign.github.io (you'll need to
request access). Point `--video_root` at the directory of per-sentence
`.mp4` clips and `--train_csv` / `--val_csv` at the corresponding split
CSVs (columns `SENTENCE_NAME`, `SENTENCE` by default — see `dataset.py`
docstring if your CSV layout differs).

## Fine-tune

```bash
python train.py \
    --train_csv /data/how2sign/train.csv \
    --val_csv   /data/how2sign/val.csv \
    --video_root /data/how2sign/clips \
    --output_dir ./checkpoints \
    --epochs 10 --batch_size 4 --lr 3e-5 --grad_accum 4
```

On a T4 (16GB), `batch_size=4` with `grad_accum=4` (effective batch 16)
and the encoder frozen should fit comfortably — most of the memory goes to
GPT-2 activations, not VideoMAE, since the encoder never needs gradients.

## Streaming inference

```bash
python stream_infer.py --checkpoint checkpoints/best.pt \
    --window_seconds 3 --stride_seconds 1.5
```

This grabs frames at `--capture_fps`, keeps a rolling `window_seconds`
buffer, translates it, then slides forward by `stride_seconds` (the
overlap between windows helps avoid chopping a sign in half at a boundary).
Tune `window_seconds` down toward 2s for lower latency or up toward 5s if
short clips are producing garbled/partial translations.

## Known rough edges to expect

- GPT-2's `generate()` with `inputs_embeds` needs a reasonably recent
  `transformers` (>=4.40); older versions silently mis-handle the prefix.
- `decord` video decoding can be flaky on some Colab T4 images — if you
  hit the same kind of import/path issues you ran into before, try
  `pip install decord==0.6.0` explicitly rather than a compiled-from-source
  build.
- Word-level streaming output will read like disconnected phrases at
  first; How2Sign fine-tunes typically need the encoder unfrozen for the
  last few epochs (small LR, e.g. 1e-6) to get fluent sentences — budget
  for that as a phase 2 once the frozen-encoder run converges.
