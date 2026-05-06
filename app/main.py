"""Streamlit UI for the surgical-video AI agent.

Run:
    streamlit run app/main.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Allow `streamlit run app/main.py` from the project root.
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.ai_clients import PROVIDERS, ProviderConfig, SURGERY_CATALOG  # noqa: E402
from app.history import HISTORY_DIR, load_index  # noqa: E402
from app.pipeline import run_analysis  # noqa: E402


st.set_page_config(page_title="DEXLab Surgical Video Agent", layout="wide")
st.title("Surgical Video AI Agent")
st.caption("Upload a surgical video, extract frames, ask a vision LLM to identify phase / instruments / organs, and keep a JSON history record.")

# ---------------------------------------------------------------------------
# Step 1 — API key linking
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Step 1 — Link API key")

    provider_key = st.selectbox(
        "Provider",
        options=list(PROVIDERS.keys()),
        format_func=lambda k: PROVIDERS[k]["label"],
    )
    pinfo = PROVIDERS[provider_key]
    st.markdown(f"[Get an API key]({pinfo['key_help']})")

    # Pre-fill from environment if available
    env_var = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
    }[provider_key]

    saved = st.session_state.get(f"key__{provider_key}", os.environ.get(env_var, ""))
    api_key = st.text_input(
        f"{pinfo['label']} API key",
        value=saved,
        type="password",
        help=f"Or set the {env_var} environment variable.",
    )
    model = st.text_input("Model", value=pinfo["default_model"])

    if api_key:
        st.session_state[f"key__{provider_key}"] = api_key
        st.success("API key linked for this session.", icon="✅")
    else:
        st.info("Paste your API key to continue.", icon="ℹ️")

    st.divider()
    st.header("Sampling")
    send_full_video = False
    if provider_key == "gemini":
        send_full_video = st.checkbox(
            "Send the FULL video to Gemini (skip frame extraction)",
            value=False,
            help="Gemini natively ingests video files via the Files API. The "
                 "model samples frames internally (~1 fps) and sees temporal "
                 "context. Anthropic and OpenAI do NOT support direct video.",
        )
    if send_full_video:
        st.caption("Full-video mode: frame count ignored.")
        num_frames = 0
    else:
        num_frames = int(st.number_input(
            "Frames to send to the model",
            min_value=1,
            value=8,
            step=1,
            help="Frames are sampled evenly across the video. No upper bound — "
                 "but more frames = more image tokens, longer latency, higher cost.",
        ))
    keep_frames = st.checkbox(
        "Keep extracted frames on disk (app/runs/...)",
        value=True,
        help="Each run is saved under app/runs/<timestamp>_<video>/ with raw "
             "frames (frames/) and the exact bytes sent to the AI (frames_sent/). "
             "Uncheck to delete the folder after analysis.",
    )

# ---------------------------------------------------------------------------
# Step 2 — Upload video
# ---------------------------------------------------------------------------
st.subheader("Step 2 — Upload a surgical video")
upload = st.file_uploader(
    "Drop a video file (mp4, mov, avi, mkv)",
    type=["mp4", "mov", "avi", "mkv", "m4v"],
    accept_multiple_files=False,
)

# ---------------------------------------------------------------------------
# Step 2.5 — Surgical context (user-supplied, narrows AI's work)
# ---------------------------------------------------------------------------
st.subheader("Step 2.5 — Surgical context")
catalog_names = list(SURGERY_CATALOG.keys())
surgery_choice = st.radio(
    "Surgery type",
    options=catalog_names + ["Other"],
    horizontal=False,
    help="Pick from the catalog, or 'Other' to enter a custom procedure name.",
)
is_other_surgery = surgery_choice == "Other"
if is_other_surgery:
    custom_name = st.text_input(
        "Custom surgery name",
        value="",
        placeholder="e.g. Laparoscopic Sleeve Gastrectomy",
    )
    surgery_type_value = f"Other: {custom_name}" if custom_name else "Other"
    default_inst: list[str] = []
    default_organs: list[str] = []
else:
    surgery_type_value = surgery_choice
    default_inst = SURGERY_CATALOG[surgery_choice]["instruments"]
    default_organs = SURGERY_CATALOG[surgery_choice]["organs_tissues"]

c_left, c_right = st.columns(2)
with c_left:
    st.markdown("**Expected instruments**")
    inst_text = st.text_area(
        "One per line — edit / add / delete freely",
        value="\n".join(default_inst),
        height=220,
        key=f"inst_text_{surgery_choice}",
        placeholder="Grasper\nClipper\nCoagulation hook\n...",
    )
    instruments_selected = [x.strip() for x in inst_text.splitlines() if x.strip()]
with c_right:
    st.markdown("**Expected organs / tissues**")
    organs_text = st.text_area(
        "One per line — edit / add / delete freely",
        value="\n".join(default_organs),
        height=220,
        key=f"organ_text_{surgery_choice}",
        placeholder="Gallbladder\nCystic duct\nLiver bed\n...",
    )
    organs_selected = [x.strip() for x in organs_text.splitlines() if x.strip()]

user_ctx = {
    "surgery_type": surgery_type_value,
    "is_other_surgery": is_other_surgery,
    "instruments": instruments_selected,
    "organs_tissues": organs_selected,
}

# ---------------------------------------------------------------------------
# Step 3 — Run
# ---------------------------------------------------------------------------
st.subheader("Step 3 — Run analysis")
run_btn = st.button(
    "Analyze video",
    type="primary",
    disabled=not (upload and api_key),
    use_container_width=True,
)

if run_btn:
    # Persist the upload to a temp file the pipeline can probe with cv2.
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix)
    tmp.write(upload.getbuffer())
    tmp.flush()
    tmp.close()
    video_path = Path(tmp.name)

    cfg = ProviderConfig(name=provider_key, api_key=api_key, model=model)

    progress_bar = st.progress(0.0, text="Starting…")
    status = st.empty()

    def _progress(label: str, frac: float):
        progress_bar.progress(min(max(frac, 0.0), 1.0), text=label)
        status.write(label)

    try:
        with st.spinner("Running pipeline…"):
            record = run_analysis(
                video_path,
                cfg,
                num_frames=num_frames if num_frames else 8,
                keep_frames=keep_frames,
                progress=_progress,
                send_full_video=send_full_video,
                user_ctx=user_ctx,
            )
        st.success(f"Saved history record: {record.get('_run_file')}")
        if record.get("run_dir"):
            st.info(
                f"Run folder: `{record['run_dir']}`  \n"
                "Contains `frames/` (raw, q=90), `frames_sent/` (exact bytes sent to AI, q=85), "
                "and `analysis.json`.",
                icon="📂",
            )
        st.session_state["last_record"] = record
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.exception(e)
    finally:
        try:
            video_path.unlink(missing_ok=True)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Result display — pick which record to render: latest run or any past run.
# ---------------------------------------------------------------------------
_history_idx = load_index()
_history_entries = list(reversed(_history_idx))  # newest first

CURRENT = "(current run)"
record = None

if _history_entries or st.session_state.get("last_record"):
    options = [CURRENT] + [
        f"{e.get('timestamp_utc', '?')} · "
        f"{e.get('video', {}).get('filename', '?')} · "
        f"{(e.get('surgery_type') or {}).get('name') or '?'} · "
        f"{(e.get('provider') or {}).get('name') or '?'}"
        for e in _history_entries
    ]
    selection = st.selectbox(
        "Show analysis for",
        options=options,
        index=0,
        help="Pick a past run to re-render its full analysis here. "
             "'(current run)' shows whatever you just analyzed.",
    )

    if selection == CURRENT:
        record = st.session_state.get("last_record")
    else:
        sel_idx = options.index(selection) - 1
        entry = _history_entries[sel_idx]
        run_file = HISTORY_DIR / entry["run_file"]
        try:
            record = json.loads(run_file.read_text(encoding="utf-8"))
        except Exception as e:
            st.error(f"Could not load {run_file}: {e}")
            record = None

if record:
    is_historical = record is not st.session_state.get("last_record")
    st.subheader("Historical analysis" if is_historical else "Latest analysis")
    if is_historical:
        st.caption(
            f"Run file: `{record.get('_run_file', '?')}`  ·  "
            f"video: `{record.get('video', {}).get('filename', '?')}`"
        )
    analysis = record.get("analysis", {})

    stype = analysis.get("surgery_type") or {}
    if stype:
        st.markdown(
            f"**Surgery type:** {stype.get('name', '—')}  "
            f"·  *confidence:* {stype.get('confidence', '—')}"
        )
        reason = stype.get("reasoning") or stype.get("evidence")  # back-compat
        if reason:
            st.caption(f"_Step 1 reasoning:_ {reason}")

    summary = analysis.get("video_summary", {})
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**Dominant phase**")
            st.write(summary.get("dominant_phase", "—"))
            st.markdown("**Phases observed**")
            st.write(summary.get("phases_observed", []) or "—")
        with c2:
            st.markdown("**Instruments observed**")
            st.write(summary.get("instruments_observed", []) or "—")
        with c3:
            st.markdown("**Organs affected**")
            st.write(
                summary.get("organs_affected")
                or summary.get("organs_observed")  # back-compat
                or "—"
            )
        with c4:
            st.markdown("**Other anatomy visible**")
            st.write(summary.get("anatomy_visible", []) or "—")
        if summary.get("narrative"):
            st.markdown("**Narrative**")
            st.write(summary["narrative"])

    frames = analysis.get("frames", [])
    if frames:
        st.markdown("**Per-frame predictions (with step reasoning)**")
        st.dataframe(
            [
                {
                    "frame_index": f.get("frame_index"),
                    "timestamp_sec": f.get("timestamp_sec"),
                    "phase": f.get("phase"),
                    "phase_reasoning": f.get("phase_reasoning", ""),
                    "instruments": ", ".join(f.get("instruments") or []),
                    "instruments_reasoning": f.get("instruments_reasoning", ""),
                    "organs_affected": ", ".join(
                        f.get("organs_affected") or f.get("organs") or []
                    ),
                    "organs_reasoning": f.get("organs_reasoning", ""),
                    "anatomy_visible": ", ".join(f.get("anatomy_visible") or []),
                    "notes": f.get("notes", ""),
                }
                for f in frames
            ],
            use_container_width=True,
            hide_index=True,
        )

    # ---- Phase timeline ----
    timeline = analysis.get("phase_timeline") or []
    if timeline:
        st.markdown("**Phase timeline**")
        df_tl = pd.DataFrame(timeline)
        # Bar chart (Gantt-like) — one horizontal bar per segment
        try:
            import altair as alt  # built-in with streamlit

            df_chart = df_tl.copy()
            df_chart["row"] = df_chart["phase"]
            chart = alt.Chart(df_chart).mark_bar().encode(
                x=alt.X("start_sec:Q", title="time (s)"),
                x2="end_sec:Q",
                y=alt.Y("phase:N", sort=None, title=None),
                color=alt.Color("phase:N", legend=None),
                tooltip=["phase", "start_sec", "end_sec", "description"],
            ).properties(height=max(120, 32 * len(df_tl)))
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            pass
        st.dataframe(df_tl, use_container_width=True, hide_index=True)

    # ---- Instrument enter/exit events ----
    events = analysis.get("instrument_events") or []
    if events:
        st.markdown("**Instrument enter / exit events**")
        st.dataframe(
            pd.DataFrame(events).sort_values("timestamp_sec").reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    # ---- Error analysis ----
    err = analysis.get("error_analysis") or {}
    if err:
        st.markdown("**Error / quality analysis**")
        if err.get("summary"):
            st.write(err["summary"])
        incidents = err.get("incidents") or []
        if incidents:
            st.dataframe(
                pd.DataFrame(incidents),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No specific incidents flagged.")

    st.markdown("**Full JSON record**")
    st.json(record, expanded=False)
    st.download_button(
        "Download this record (.json)",
        data=json.dumps(record, indent=2, ensure_ascii=False),
        file_name=Path(record.get("_run_file", "analysis.json")).name,
        mime="application/json",
    )

# ---------------------------------------------------------------------------
# History panel
# ---------------------------------------------------------------------------
with st.expander("Run history", expanded=False):
    idx = load_index()
    if not idx:
        st.write("No prior runs yet.")
    else:
        st.write(f"{len(idx)} prior run(s).")
        st.dataframe(
            [
                {
                    "timestamp_utc": e.get("timestamp_utc"),
                    "video": e.get("video", {}).get("filename"),
                    "surgery_type": (e.get("surgery_type") or {}).get("name"),
                    "provider": e.get("provider", {}).get("name"),
                    "model": e.get("provider", {}).get("model"),
                    "dominant_phase": (e.get("summary") or {}).get("dominant_phase"),
                    "run_file": e.get("run_file"),
                }
                for e in reversed(idx)
            ],
            use_container_width=True,
            hide_index=True,
        )
