"""Video frame extraction for the surgical-analysis agent."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2


@dataclass
class Frame:
    index: int                          # original frame index in the source video
    timestamp: float                    # seconds from start
    path: Path                          # raw on-disk JPEG q=90 (for inspection)
    width: int                          # original frame width
    height: int                         # original frame height
    sent_path: Optional[Path] = None    # exact bytes sent to AI on disk (inspection)
    sent_bytes: Optional[bytes] = None  # exact bytes sent to AI, in memory
    sent_width: Optional[int] = None    # width after optional resize
    sent_height: Optional[int] = None   # height after optional resize

    def to_base64(self, max_side: int = 1568, quality: int = 92) -> str:
        """Return the API-ready base64 string.

        Fast path: if `sent_bytes` is already cached (set by extract_frames at
        decode time), just base64-encode it — no disk read, no re-encode.

        Fallback (no cache): re-read raw JPEG from disk, optionally resize,
        re-encode at the requested quality. Kept for backwards compatibility.
        """
        if self.sent_bytes is not None:
            return base64.b64encode(self.sent_bytes).decode("ascii")

        img = cv2.imread(str(self.path))
        if img is None:
            raise RuntimeError(f"Failed to read frame {self.path}")
        h, w = img.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        jpeg_bytes = buf.tobytes()
        if self.sent_path is not None:
            self.sent_path.parent.mkdir(parents=True, exist_ok=True)
            self.sent_path.write_bytes(jpeg_bytes)
        # Cache for any subsequent calls.
        self.sent_bytes = jpeg_bytes
        return base64.b64encode(jpeg_bytes).decode("ascii")


@dataclass
class VideoInfo:
    path: Path
    fps: float
    total_frames: int
    width: int
    height: int
    duration_sec: float


def probe(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = total / fps if fps > 0 else 0.0
    return VideoInfo(video_path, fps, total, w, h, duration)


def extract_frames(
    video_path: Path,
    out_dir: Path,
    num_frames: int = 12,
    progress_cb=None,
    sent_dir: Optional[Path] = None,
    max_side: int = 1568,
    sent_quality: int = 92,
    raw_quality: int = 95,
) -> tuple[VideoInfo, List[Frame]]:
    """Sample `num_frames` evenly-spaced frames from a video.

    Per frame, this function decodes ONCE from the video, then immediately:
      1. Writes a "raw" inspection copy at original resolution to ``out_dir``
         (JPEG q=raw_quality).
      2. Encodes the API-ready JPEG (q=sent_quality, optionally resized so the
         longest side <= max_side). Those bytes are:
           - cached on the Frame object (Frame.sent_bytes) for zero-disk-roundtrip
             base64 encoding by the AI clients;
           - written to ``sent_dir/<same name>`` if sent_dir is given, so the user
             can inspect EXACTLY what the API received.

    The first and last frames are always included when num_frames >= 2.
    """
    info = probe(video_path)
    if info.total_frames <= 0:
        raise RuntimeError("Video has zero readable frames")

    out_dir.mkdir(parents=True, exist_ok=True)
    if sent_dir is not None:
        sent_dir.mkdir(parents=True, exist_ok=True)

    n = max(1, min(num_frames, info.total_frames))
    if n == 1:
        targets = [info.total_frames // 2]
    else:
        step = (info.total_frames - 1) / (n - 1)
        targets = [int(round(i * step)) for i in range(n)]

    cap = cv2.VideoCapture(str(video_path))
    frames: List[Frame] = []
    try:
        for i, idx in enumerate(targets):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, img = cap.read()
            if not ok or img is None:
                continue
            ts = idx / info.fps if info.fps > 0 else 0.0
            fname = f"frame_{i:03d}_idx{idx}.jpg"

            # 1. Raw inspection copy (original resolution).
            fp = out_dir / fname
            cv2.imwrite(str(fp), img, [cv2.IMWRITE_JPEG_QUALITY, raw_quality])

            # 2. Encode the API-ready version once, from the in-memory array.
            h0, w0 = img.shape[:2]
            scale = min(1.0, max_side / max(h0, w0))
            if scale < 1.0:
                sent_img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA
                )
            else:
                sent_img = img
            ok_enc, buf = cv2.imencode(
                ".jpg", sent_img, [cv2.IMWRITE_JPEG_QUALITY, sent_quality]
            )
            if not ok_enc:
                raise RuntimeError(f"JPEG encode failed for frame {idx}")
            sent_bytes = buf.tobytes()
            sh, sw = sent_img.shape[:2]

            sp: Optional[Path] = None
            if sent_dir is not None:
                sp = sent_dir / fname
                sp.write_bytes(sent_bytes)

            frames.append(Frame(
                index=idx,
                timestamp=ts,
                path=fp,
                width=w0,
                height=h0,
                sent_path=sp,
                sent_bytes=sent_bytes,
                sent_width=sw,
                sent_height=sh,
            ))
            if progress_cb:
                progress_cb(i + 1, len(targets))
    finally:
        cap.release()

    if not frames:
        raise RuntimeError("Failed to extract any frames")
    return info, frames
