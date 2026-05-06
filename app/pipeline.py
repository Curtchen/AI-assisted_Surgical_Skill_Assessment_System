"""Orchestrator: video -> frames -> AI -> structured analysis -> history record."""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .ai_clients import ProviderConfig, analyze, analyze_video_with_gemini
from .history import save_run
from .video_utils import extract_frames, probe


ProgressFn = Callable[[str, float], None]
# progress(stage_label, fraction_in_[0,1])

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "video"


def run_analysis(
    video_path: Path,
    provider_cfg: ProviderConfig,
    num_frames: int = 12,
    workdir: Optional[Path] = None,
    keep_frames: bool = True,
    progress: Optional[ProgressFn] = None,
    send_full_video: bool = False,
    user_ctx: Optional[dict] = None,
) -> dict:
    """Full pipeline. Returns the saved record (also persisted to history).

    By default each run is saved under ``app/runs/<timestamp>_<video_stem>/``
    with three artifacts:
      - frames/        original-resolution JPEG q=90 (the "raw" disk copy)
      - frames_sent/   exact bytes sent to the AI (JPEG q=85, possibly resized)
      - analysis.json  the full record (also indexed in app/history/history.json)

    If ``keep_frames`` is False, the run dir is deleted after the record is saved
    (the history JSON files in app/history/ still survive).
    """
    if workdir is None:
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_safe(video_path.stem)}"
        workdir = RUNS_DIR / run_id
    workdir.mkdir(parents=True, exist_ok=True)
    # Let providers dump raw responses to this run dir on JSON parse failure.
    provider_cfg.dump_dir = workdir

    try:
        if progress:
            progress("Probing video", 0.05)
        info = probe(video_path)

        if progress:
            progress("Extracting frames", 0.10)

        def _frame_cb(done, total):
            if progress:
                progress(
                    f"Extracting frames ({done}/{total})",
                    0.10 + 0.30 * (done / total),
                )

        if send_full_video:
            # Bypass frame extraction; let the model ingest the full video.
            if provider_cfg.name != "gemini":
                raise RuntimeError(
                    "send_full_video=True is only supported for the Gemini provider; "
                    "Anthropic and OpenAI APIs do not accept video files directly."
                )
            frames = []  # nothing to enumerate

            def _gem_cb(label: str):
                if progress:
                    progress(label, 0.55)

            analysis = analyze_video_with_gemini(
                provider_cfg, video_path, progress_cb=_gem_cb, user_ctx=user_ctx
            )
        else:
            frames_dir = workdir / "frames"
            sent_dir = workdir / "frames_sent"
            info, frames = extract_frames(
                video_path,
                frames_dir,
                num_frames=num_frames,
                progress_cb=_frame_cb,
                sent_dir=sent_dir,
            )

            if progress:
                progress(f"Calling {provider_cfg.name} ({provider_cfg.model})", 0.45)
            analysis = analyze(provider_cfg, frames, user_ctx=user_ctx)

        if progress:
            progress("Saving history", 0.92)

        record = {
            "video": {
                "filename": video_path.name,
                "path": str(video_path),
                "fps": info.fps,
                "total_frames": info.total_frames,
                "duration_sec": info.duration_sec,
                "width": info.width,
                "height": info.height,
            },
            "sampling": {
                "mode": "full_video" if send_full_video else "frame_extraction",
                "num_frames_requested": num_frames,
                "num_frames_used": len(frames),
                "frame_indices": [f.index for f in frames],
                "frame_timestamps_sec": [round(f.timestamp, 3) for f in frames],
            },
            "provider": {
                "name": provider_cfg.name,
                "model": provider_cfg.model,
            },
            "user_context": user_ctx or {},
            "run_dir": str(workdir),
            "frames_raw": [str(f.path) for f in frames],
            "frames_sent": [str(f.sent_path) if f.sent_path else None for f in frames],
            "analysis": analysis,
        }

        # Per-run artifact alongside the frames.
        local_record_path = workdir / "analysis.json"
        local_record_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Also append to the rolling history index.
        run_path = save_run(record)
        record["_run_file"] = str(run_path)
        record["_local_record"] = str(local_record_path)

        if progress:
            progress("Done", 1.0)
        return record
    finally:
        if not keep_frames:
            shutil.rmtree(workdir, ignore_errors=True)
