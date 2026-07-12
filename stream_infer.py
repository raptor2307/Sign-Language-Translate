"""
Live streaming ASL -> English inference.

Accumulates a rolling buffer of webcam frames. Every WINDOW_SECONDS of
footage, it samples num_frames uniformly from the current window, runs the
model, prints the translation, then slides the window forward by
STRIDE_SECONDS (keeping some overlap so a sign split across window
boundaries still has a decent chance of appearing whole in the next window).

Usage:
    python stream_infer.py --checkpoint checkpoints/best.pt --window_seconds 3 --stride_seconds 1.5

Press 'q' to quit.
"""

import argparse
import collections
import time

import cv2
import torch

from model import SignTranslationModel, build_image_processor


class SlidingWindowBuffer:
    def __init__(self, fps: int, window_seconds: float, stride_seconds: float):
        self.window_frames = int(fps * window_seconds)
        self.stride_frames = int(fps * stride_seconds)
        self.buffer = collections.deque(maxlen=self.window_frames)

    def add(self, frame):
        self.buffer.append(frame)

    def ready(self) -> bool:
        return len(self.buffer) == self.window_frames

    def get_window(self):
        return list(self.buffer)

    def slide(self):
        # Drop the oldest `stride_frames` so the next window overlaps the
        # tail of this one instead of starting from scratch.
        for _ in range(min(self.stride_frames, len(self.buffer))):
            self.buffer.popleft()


def sample_uniform(frames, k):
    if len(frames) <= k:
        return frames
    idx = [round(i * (len(frames) - 1) / (k - 1)) for i in range(k)]
    return [frames[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--camera_index", type=int, default=0)
    ap.add_argument("--capture_fps", type=int, default=15)
    ap.add_argument("--window_seconds", type=float, default=3.0)
    ap.add_argument("--stride_seconds", type=float, default=1.5)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--num_visual_tokens", type=int, default=16)
    ap.add_argument("--gpt2_name", default="gpt2")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SignTranslationModel(
        gpt2_name=args.gpt2_name,
        num_visual_tokens=args.num_visual_tokens,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    image_processor = build_image_processor()

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FPS, args.capture_fps)

    win = SlidingWindowBuffer(args.capture_fps, args.window_seconds, args.stride_seconds)

    frame_interval = 1.0 / args.capture_fps
    last_capture = 0.0
    last_translation = ""

    print("Streaming... press 'q' to quit.")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            now = time.time()
            if now - last_capture >= frame_interval:
                last_capture = now
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                win.add(frame_rgb)

                if win.ready():
                    clip = sample_uniform(win.get_window(), args.num_frames)
                    pixel_values = image_processor(clip, return_tensors="pt")["pixel_values"].to(device)

                    with torch.no_grad():
                        translations = model.generate(pixel_values, max_new_tokens=30, num_beams=1)
                    last_translation = translations[0]
                    print(f"[{time.strftime('%H:%M:%S')}] {last_translation}")

                    win.slide()

            # overlay + show
            display = frame_bgr.copy()
            cv2.putText(display, last_translation, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("ASL -> English (streaming)", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
