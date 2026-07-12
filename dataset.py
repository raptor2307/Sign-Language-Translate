"""
How2Sign dataset loader.

Expects the How2Sign "re-segmented" clip CSVs (the ones distributed with
columns SENTENCE_NAME, SENTENCE) and a directory of per-sentence clip
files named "<SENTENCE_NAME>.mp4" (this matches how How2Sign clips are
normally shipped after segmentation). If your CSV already has explicit
clip paths, just make sure it has a "clip_path" and "sentence" column
and pass clip_path_col / sentence_col accordingly.

Example CSV row (How2Sign val split):
    VIDEO_ID, VIDEO_NAME, SENTENCE_ID, SENTENCE_NAME, START, END, SENTENCE
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu

decord = None  # keep decord import lazy-friendly for environments w/o GPU decord


class How2SignDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        video_root: str,
        tokenizer,
        num_frames: int = 16,
        frame_size: int = 224,
        max_text_len: int = 64,
        clip_path_col: str = None,
        sentence_col: str = "SENTENCE",
        sentence_name_col: str = "SENTENCE_NAME",
        video_ext: str = ".mp4",
    ):
        self.df = pd.read_csv(csv_path, sep="\t" if csv_path.endswith(".tsv") else ",")
        self.video_root = video_root
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.max_text_len = max_text_len
        self.clip_path_col = clip_path_col
        self.sentence_col = sentence_col
        self.sentence_name_col = sentence_name_col
        self.video_ext = video_ext

        # Drop rows whose video file doesn't actually exist on disk
        keep = []
        for i, row in self.df.iterrows():
            if os.path.exists(self._clip_path(row)):
                keep.append(i)
        dropped = len(self.df) - len(keep)
        if dropped:
            print(f"[How2SignDataset] Skipping {dropped} rows with missing video files.")
        self.df = self.df.loc[keep].reset_index(drop=True)

    def _clip_path(self, row) -> str:
        if self.clip_path_col:
            return os.path.join(self.video_root, row[self.clip_path_col])
        return os.path.join(self.video_root, f"{row[self.sentence_name_col]}{self.video_ext}")

    def __len__(self):
        return len(self.df)

    def _sample_frames(self, path: str) -> np.ndarray:
        vr = VideoReader(path, ctx=cpu(0))
        total = len(vr)
        if total == 0:
            raise RuntimeError(f"Empty video: {path}")
        # Uniform sampling across the whole clip, matching VideoMAE's
        # standard 16-frame tubelet sampling.
        idx = np.linspace(0, total - 1, num=self.num_frames).astype(np.int64)
        frames = vr.get_batch(idx).asnumpy()  # (T, H, W, 3) uint8
        return frames

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = self._clip_path(row)
        frames = self._sample_frames(path)  # (T, H, W, 3)

        # Resize done here (cheap) rather than pulling in a full transforms
        # dependency; VideoMAEImageProcessor in train.py handles normalization.
        sentence = str(row[self.sentence_col]).strip()

        tok = self.tokenizer(
            sentence,
            max_length=self.max_text_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "frames": frames,                       # raw uint8 (T,H,W,3), processed in collate
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "sentence": sentence,
        }


def make_collate_fn(image_processor):
    """
    Returns a collate_fn that runs VideoMAEImageProcessor over the batch of
    raw frame arrays (handles resize/normalize/stack) and stacks the
    tokenized text.
    """

    def collate(batch):
        frame_list = [b["frames"] for b in batch]  # list of (T,H,W,3)
        pixel_values = image_processor(
            list(frame_list), return_tensors="pt"
        )["pixel_values"]  # (B, T, C, H, W)

        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        sentences = [b["sentence"] for b in batch]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sentences": sentences,
        }

    return collate
