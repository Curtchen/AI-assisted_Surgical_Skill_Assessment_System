# DEXLab Surgical Video Agent

Streamlit UI that walks the user through:

1. **Linking an API key** — Anthropic (Claude), OpenAI, or Google (Gemini).
2. **Uploading a surgical video** (mp4/mov/avi/mkv).
3. **Frame extraction** — N evenly-spaced frames are written to a temp dir.
4. **Vision-LLM analysis** — frames are sent to the chosen model with a strict JSON schema asking for per-frame *phase / instruments / organs* and a video-level summary.
5. **History record** — every run is saved as a standalone JSON under `app/history/run_*.json` and indexed in `app/history/history.json`.

## Run

```bash
pip install -r app/requirements.txt
streamlit run app/main.py
```

Then open the URL Streamlit prints (default <http://localhost:8501>).

## Module layout

| File | Role |
| --- | --- |
| `main.py` | Streamlit UI |
| `pipeline.py` | Orchestrator: probe → extract frames → call AI → save record |
| `video_utils.py` | OpenCV-based frame sampler |
| `ai_clients.py` | Pluggable Anthropic / OpenAI / Gemini vision clients |
| `history.py` | Per-run JSON files + rolling history index |

## Output schema

```json
{
  "video": {"filename": "...", "fps": 30.0, "duration_sec": 60.0, ...},
  "sampling": {"num_frames_used": 8, "frame_indices": [...], ...},
  "provider": {"name": "anthropic", "model": "claude-opus-4-5"},
  "analysis": {
    "video_summary": {
      "dominant_phase": "...",
      "phases_observed": ["..."],
      "instruments_observed": ["..."],
      "organs_observed": ["..."],
      "narrative": "..."
    },
    "frames": [
      {"frame_index": 0, "timestamp_sec": 0.0, "phase": "...",
       "instruments": ["..."], "organs": ["..."], "notes": "..."}
    ]
  }
}
```

## Notes

- API keys are kept in Streamlit `session_state` only; they are not written to disk. You can also pre-set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY` in your environment.
- Frames are JPEG-encoded and downscaled to a max side of 1024 px before being base64-embedded in the request.
- The model is instructed to emit JSON only; the parser tolerates accidental code fences.
