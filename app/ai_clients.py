"""Vision-LLM clients for surgical scene analysis.

Each provider exposes the same `analyze(frames, system_prompt, user_prompt) -> dict`
interface, returning a JSON-decoded structured response.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from .video_utils import Frame


# ---------------------------------------------------------------------------
# Surgery catalog: known procedure types and their controlled vocabularies.
# Add a new procedure here and the system prompt updates automatically.
# ---------------------------------------------------------------------------
SURGERY_CATALOG: dict[str, dict[str, list[str]]] = {
    "Laparoscopic Cholecystectomy": {
        "phases": [
            "Preparation",
            "Calot's triangle dissection",
            "Clipping and cutting",
            "Gallbladder dissection",
            "Gallbladder packaging",
            "Cleaning and coagulation",
            "Gallbladder retraction",
        ],
        "instruments": [
            "Grasper", "Clipper", "Coagulation hook", "Bipolar",
            "Scissors", "Suction-irrigation", "Specimen bag", "Stapler",
        ],
        "organs_tissues": [
            "Gallbladder", "Cystic duct", "Cystic artery",
            "Calot's triangle", "Liver bed", "Hartmann's pouch",
            "Common bile duct", "Peritoneum", "Omentum",
        ],
    },
    "Laparoscopic Appendectomy": {
        "phases": [
            "Trocar placement and exposure",
            "Adhesiolysis",
            "Mesoappendix dissection",
            "Appendix base ligation",
            "Appendix transection",
            "Specimen retrieval",
            "Hemostasis and closure",
        ],
        "instruments": [
            "Grasper", "Endo-loop", "Clipper", "Coagulation hook",
            "Bipolar", "Scissors", "Stapler", "Suction-irrigation", "Specimen bag",
        ],
        "organs_tissues": [
            "Appendix", "Mesoappendix", "Cecum", "Terminal ileum",
            "Peritoneum", "Omentum",
        ],
    },
    "Laparoscopic Inguinal Hernia Repair (TAPP/TEP)": {
        "phases": [
            "Peritoneal flap creation",
            "Hernia sac reduction",
            "Mesh placement",
            "Mesh fixation",
            "Peritoneal closure",
        ],
        "instruments": [
            "Grasper", "Scissors", "Coagulation hook", "Bipolar",
            "Mesh", "Tacker / Stapler", "Suction-irrigation", "Suture",
        ],
        "organs_tissues": [
            "Peritoneum", "Hernia sac", "Inguinal canal", "Spermatic cord",
            "Iliac vessels", "Vas deferens", "Cooper's ligament",
        ],
    },
}


def _format_catalog(catalog: dict[str, dict[str, list[str]]]) -> str:
    blocks = []
    for proc, voc in catalog.items():
        blocks.append(f"### {proc}")
        blocks.append(f"  Phases: {', '.join(voc['phases'])}")
        blocks.append(f"  Instruments: {', '.join(voc['instruments'])}")
        blocks.append(f"  Organs / tissues affected: {', '.join(voc['organs_tissues'])}")
        blocks.append("")
    return "\n".join(blocks)


SCHEMA_INSTRUCTION = f"""You are a board-certified laparoscopic surgeon with 15+ years
of experience in hepato-pancreato-biliary and minimally invasive general surgery, and a
recognized expert in surgical video review for skill assessment and intra-operative
quality control. You routinely identify procedures, phases, instruments, and anatomy
from endoscopic footage and articulate the visual cues that justify each call.

Approach this task as a senior reviewer would: precise vocabulary, no hand-waving, and
explicit visual evidence at every step. If a frame is ambiguous, say so — do not invent
detail you cannot see.

You are shown a temporally ordered sequence of frames sampled from ONE surgical video.
Reason in three sequential steps and report your reasoning at every step.

============================================================
SURGERY CATALOG — controlled vocabulary per procedure
============================================================
{_format_catalog(SURGERY_CATALOG)}

============================================================
STEP 1 — Identify the SURGERY TYPE
============================================================
Look across ALL frames as a whole. Pick exactly one surgery type from the catalog
above, or output "Other: <free-form name>" if no entry fits.
You MUST justify the choice in `surgery_type.reasoning` by citing concrete visual
cues (visible anatomy, characteristic instruments, port layout, color/texture of
the operative field, etc.).

============================================================
STEP 2 — For EACH frame, identify the SURGICAL PHASE
============================================================
Using ONLY the phases listed for the surgery type chosen in Step 1, label each
frame's current phase. (If you chose "Other: ...", free-form phase labels are OK.)
For every frame, fill `phase_reasoning` with a one-sentence justification: what in
THIS frame indicates that phase (e.g. "clip applier engaging cystic duct" → Clipping
and cutting). If the frame is out-of-body, blurry, or occluded, set
`phase = "Out of body / unclear"` and explain in `phase_reasoning`.

============================================================
STEP 3 — For each frame, identify INSTRUMENTS and ORGANS
============================================================
For every frame, fill BOTH lists using only the catalog's vocabulary for the chosen
surgery type (or free-form if surgery type is "Other"):
  - `instruments`: instruments visible AND in use in this frame
  - `organs_affected`: tissues being actively operated on (cut, dissected, clipped,
    grasped, cauterized, retracted) in this frame
  - `anatomy_visible`: other anatomy visible in the background but NOT operated on
You MUST justify these choices in:
  - `instruments_reasoning`: visual cues identifying each instrument (jaw shape,
    color, action being performed)
  - `organs_reasoning`: why these specific tissues are considered "affected" right
    now (what is the instrument doing to them)

If nothing is visible / in use, leave the list empty AND say so in the reasoning.

============================================================
OUTPUT — Return ONLY valid JSON matching this schema
============================================================
No prose outside JSON. No markdown fences.

{{
  "surgery_type": {{
    "name": "<catalog name, or 'Other: <free-form name>'>",
    "confidence": "high | medium | low",
    "reasoning": "<why this surgery type — cite visual cues across the clip>"
  }},
  "video_summary": {{
    "dominant_phase": "<string>",
    "phases_observed": ["<string>", ...],
    "instruments_observed": ["<string>", ...],
    "organs_affected": ["<string>", ...],
    "anatomy_visible": ["<string>", ...],
    "narrative": "<one short paragraph describing what happens across the clip>"
  }},
  "frames": [
    {{
      "frame_index": <int, original video frame index>,
      "timestamp_sec": <float>,
      "phase": "<string>",
      "phase_reasoning": "<one sentence: what in this frame indicates that phase>",
      "instruments": ["<string>", ...],
      "instruments_reasoning": "<one sentence: visual cues identifying each tool>",
      "organs_affected": ["<string>", ...],
      "organs_reasoning": "<one sentence: why these tissues are being acted on now>",
      "anatomy_visible": ["<string>", ...],
      "notes": "<short string, optional>"
    }}
  ],

  "phase_timeline": [
    {{
      "start_sec": <float>,
      "end_sec": <float>,
      "phase": "<string>",
      "description": "<one sentence: what was accomplished in this segment>"
    }}
  ],

  "instrument_events": [
    {{
      "timestamp_sec": <float>,
      "frame_index": <int, nearest sampled frame>,
      "instrument": "<string>",
      "event": "enter | exit",
      "phase": "<the phase active when the event happens>"
    }}
  ],

  "error_analysis": {{
    "summary": "<one short paragraph: overall assessment of mistakes, near-misses, "
               "or suboptimal technique observed across the clip. If nothing notable, "
               "say so explicitly>",
    "incidents": [
      {{
        "timestamp_sec": <float>,
        "frame_index": <int, nearest sampled frame>,
        "phase": "<phase active at the incident>",
        "severity": "minor | moderate | severe",
        "category": "<short label, e.g. 'unintended bleeding', 'instrument clash', "
                    "'wrong-plane dissection', 'retraction loss', 'tissue tear'>",
        "description": "<2-3 sentences: what happened, what should have happened, "
                       "consequence>"
      }}
    ]
  }}
}}

============================================================
EXTRA INSTRUCTIONS FOR THE NEW FIELDS
============================================================
- `phase_timeline`: derive the segment boundaries from your per-frame phase
  labels. Use the timestamps of the FIRST and LAST sampled frame in each
  contiguous run of the same phase as start_sec / end_sec.
- `instrument_events`: scan the per-frame `instruments` lists in temporal order.
  When an instrument appears that wasn't in the previous frame, emit an "enter"
  event at that frame's timestamp. When one disappears, emit an "exit". Skip
  spurious one-frame flickers if they look like miscalls.
- `error_analysis.incidents`: include observations such as bleeding without
  immediate hemostasis, gallbladder perforation, clip mis-application, wrong
  anatomical plane (CBD vs cystic duct confusion), repeated tissue grasping,
  dropped/lost instruments, view-blocking by smoke. Be specific about WHY each
  is suboptimal. If no incidents are visible, return an empty list and say so
  in `summary`.
"""


@dataclass
class ProviderConfig:
    name: str        # "anthropic" | "openai" | "gemini"
    api_key: str
    model: str
    # If set, raw LLM responses that fail JSON parsing get dumped here.
    dump_dir: Optional[Path] = None


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        if text.rstrip().endswith("```"):
            text = text.rstrip().rstrip("`").rstrip()
    return text


def _repair_json(text: str) -> str:
    """Best-effort fixes for the common ways an LLM can break a JSON dump."""
    # Remove trailing commas before } or ]
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    # Remove a trailing comma right before EOF
    text = re.sub(r",\s*\Z", "", text)
    return text


def _extract_json(text: str, dump_dir: Optional[Path] = None) -> dict:
    """Parse JSON from a possibly-noisy LLM response.

    On failure, dumps the raw text to ``dump_dir/last_response.txt`` (if given)
    and raises a JSONDecodeError annotated with the offending neighborhood.
    """
    body = _strip_fence(text)

    # Strategy 1: parse as-is.
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass

    # Strategy 2: grab the outermost {...} and try.
    m = re.search(r"\{.*\}", body, re.DOTALL)
    candidate = m.group(0) if m else body

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Strategy 3: light syntactic repair (trailing commas).
    try:
        return json.loads(_repair_json(candidate))
    except json.JSONDecodeError as e:
        if dump_dir is not None:
            try:
                dump_dir.mkdir(parents=True, exist_ok=True)
                (dump_dir / "last_response.txt").write_text(text, encoding="utf-8")
            except Exception:
                pass
        # Annotate the error with a snippet around the failure point.
        ctx_start = max(0, e.pos - 80)
        ctx_end = min(len(candidate), e.pos + 80)
        snippet = candidate[ctx_start:ctx_end].replace("\n", "\\n")
        raise json.JSONDecodeError(
            f"{e.msg} — near char {e.pos}: ...{snippet}...",
            e.doc, e.pos,
        ) from None


def _build_user_context_block(user_ctx: Optional[dict]) -> str:
    """Render the user-supplied surgical context as a hint block.

    `user_ctx` may be None or have any of:
        surgery_type        str
        instruments         list[str]
        organs_tissues      list[str]
        is_other_surgery    bool   (True if user picked "Other" — relax catalog)
    """
    if not user_ctx:
        return ""
    lines = ["", "============================================================",
             "USER-SUPPLIED CONTEXT  (treat as ground truth unless visual evidence",
             "strongly contradicts it)",
             "============================================================"]
    st = user_ctx.get("surgery_type")
    if st:
        lines.append(f"- Surgery type: {st}")
    if user_ctx.get("is_other_surgery"):
        lines.append(
            "  (User selected 'Other' — the catalog vocabulary is a hint only; "
            "use free-form labels matching this procedure.)"
        )
    inst = user_ctx.get("instruments") or []
    if inst:
        lines.append(f"- Instruments expected to appear: {', '.join(inst)}")
        lines.append("  When you see an instrument from this list, prefer this "
                     "exact label over a synonym.")
    organs = user_ctx.get("organs_tissues") or []
    if organs:
        lines.append(f"- Organs / tissues expected to be affected: {', '.join(organs)}")
        lines.append("  Use these exact labels when scoring `organs_affected`.")
    lines.append("")
    return "\n".join(lines)


def _build_user_prompt(frames: Sequence[Frame], user_ctx: Optional[dict] = None) -> str:
    lines = [
        "Sampled frames from a surgical video, in temporal order:",
    ]
    for i, f in enumerate(frames):
        lines.append(
            f"- Image {i + 1}: original_frame_index={f.index}, timestamp_sec={f.timestamp:.2f}"
        )
    lines.append(_build_user_context_block(user_ctx))
    lines.append(
        "\nReturn the JSON described in the system message. The frames array MUST contain "
        "one entry per image above, with frame_index and timestamp_sec matching the values listed."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Anthropic / Claude
# ---------------------------------------------------------------------------
def analyze_with_anthropic(
    cfg: ProviderConfig,
    frames: List[Frame],
    use_thinking: bool = True,
    thinking_budget: int = 8000,
    user_ctx: Optional[dict] = None,
) -> dict:
    """Call Claude with the frames. Extended thinking is enabled by default
    (Claude 4.x family) so the model can deliberate before emitting JSON."""
    try:
        import anthropic  # type: ignore
    except ImportError as e:
        raise RuntimeError("anthropic SDK not installed. pip install anthropic") from e

    client = anthropic.Anthropic(api_key=cfg.api_key)
    content = []
    for f in frames:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": f.to_base64(),
            },
        })
    content.append({"type": "text", "text": _build_user_prompt(frames, user_ctx)})

    # Anthropic requires max_tokens. Set it high enough to cover thinking +
    # full JSON answer; Claude 4.x accepts up to 64k output tokens.
    kwargs = dict(
        model=cfg.model,
        max_tokens=32000,
        system=SCHEMA_INSTRUCTION,
        messages=[{"role": "user", "content": content}],
    )
    if use_thinking:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

    # Anthropic refuses non-streaming requests whose worst-case duration may
    # exceed 10 minutes (high max_tokens + extended thinking can hit that).
    # Use streaming and assemble the final message.
    def _stream_call(call_kwargs: dict):
        with client.messages.stream(**call_kwargs) as stream:
            return stream.get_final_message()

    try:
        msg = _stream_call(kwargs)
    except anthropic.BadRequestError as e:
        # Common fallbacks:
        # 1. Older Claude 3.x models don't accept `thinking` at all.
        # 2. Some models cap max_tokens lower than 32k without a beta header.
        msg_text = str(e).lower()
        retried = False
        if use_thinking and "thinking" in msg_text:
            kwargs.pop("thinking", None)
            retried = True
        if "max_tokens" in msg_text and ("too large" in msg_text or "exceed" in msg_text):
            kwargs["max_tokens"] = 8192
            retried = True
        if not retried:
            raise
        msg = _stream_call(kwargs)

    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    return _extract_json(text, dump_dir=cfg.dump_dir)


# ---------------------------------------------------------------------------
# OpenAI / GPT-4o family
# ---------------------------------------------------------------------------
def analyze_with_openai(
    cfg: ProviderConfig,
    frames: List[Frame],
    user_ctx: Optional[dict] = None,
) -> dict:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError("openai SDK not installed. pip install openai") from e

    client = OpenAI(api_key=cfg.api_key)
    content = []
    for f in frames:
        b64 = f.to_base64()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": _build_user_prompt(frames, user_ctx)})

    # OpenAI: max_tokens / max_completion_tokens are OPTIONAL. We simply
    # don't pass them so the model uses its full available budget (the
    # remaining context window after the input).
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": SCHEMA_INSTRUCTION},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
    )
    return _extract_json(resp.choices[0].message.content or "", dump_dir=cfg.dump_dir)


# ---------------------------------------------------------------------------
# Google / Gemini
# ---------------------------------------------------------------------------
def analyze_with_gemini(
    cfg: ProviderConfig,
    frames: List[Frame],
    user_ctx: Optional[dict] = None,
) -> dict:
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "google-generativeai not installed. pip install google-generativeai"
        ) from e

    genai.configure(api_key=cfg.api_key)
    model = genai.GenerativeModel(
        cfg.model,
        system_instruction=SCHEMA_INSTRUCTION,
    )
    parts = []
    for f in frames:
        parts.append({"mime_type": "image/jpeg", "data": f.to_base64()})
    parts.append(_build_user_prompt(frames, user_ctx))
    resp = model.generate_content(
        parts,
        generation_config={"response_mime_type": "application/json"},
    )
    return _extract_json(resp.text, dump_dir=cfg.dump_dir)


def analyze_video_with_gemini(
    cfg: ProviderConfig,
    video_path,
    progress_cb=None,
    user_ctx: Optional[dict] = None,
) -> dict:
    """Send the FULL video file directly to Gemini (no frame extraction).

    Gemini natively ingests video files via the Files API. The model samples
    frames internally (default ~1 fps) and reasons over temporal context the
    frame-extraction path discards.

    Anthropic and OpenAI Chat Completions do NOT accept video files in their
    public APIs as of this writing — for those providers you must keep using
    the frame-extraction path (`analyze`).
    """
    import time
    from pathlib import Path

    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "google-generativeai not installed. pip install google-generativeai"
        ) from e

    genai.configure(api_key=cfg.api_key)

    if progress_cb:
        progress_cb("Uploading video to Gemini Files API")
    video_file = genai.upload_file(path=str(Path(video_path)))

    # Wait for the file to finish ACTIVE processing.
    while video_file.state.name == "PROCESSING":
        if progress_cb:
            progress_cb("Gemini is processing the video…")
        time.sleep(2)
        video_file = genai.get_file(video_file.name)
    if video_file.state.name != "ACTIVE":
        raise RuntimeError(f"Gemini file upload failed: state={video_file.state.name}")

    if progress_cb:
        progress_cb("Calling Gemini on uploaded video")
    model = genai.GenerativeModel(
        cfg.model,
        system_instruction=SCHEMA_INSTRUCTION,
    )
    user_prompt = (
        "The attached file is a SINGLE surgical video clip. Apply the three-step "
        "reasoning chain from the system instructions and emit one JSON object. "
        "Treat the entire clip as the input — for the `frames` array, sample "
        "around 8-16 representative timepoints across the clip and use those as "
        "your per-frame entries (set `frame_index` to the source-video frame "
        "index when known, otherwise to a 0-based ordinal, and `timestamp_sec` "
        "to your best estimate of the time within the clip)."
        + _build_user_context_block(user_ctx)
    )
    resp = model.generate_content(
        [video_file, user_prompt],
        generation_config={"response_mime_type": "application/json"},
    )

    # Best-effort cleanup so the user's File quota stays clean.
    try:
        genai.delete_file(video_file.name)
    except Exception:
        pass

    return _extract_json(resp.text, dump_dir=cfg.dump_dir)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
PROVIDERS = {
    "anthropic": {
        "label": "Anthropic (Claude)",
        "default_model": "claude-opus-4-5",
        "fn": analyze_with_anthropic,
        "key_help": "https://console.anthropic.com/settings/keys",
    },
    "openai": {
        "label": "OpenAI (GPT-4o / GPT-5)",
        "default_model": "gpt-4o",
        "fn": analyze_with_openai,
        "key_help": "https://platform.openai.com/api-keys",
    },
    "gemini": {
        "label": "Google (Gemini)",
        "default_model": "gemini-2.0-flash",
        "fn": analyze_with_gemini,
        "key_help": "https://aistudio.google.com/app/apikey",
    },
}


def analyze(
    cfg: ProviderConfig,
    frames: List[Frame],
    user_ctx: Optional[dict] = None,
) -> dict:
    if cfg.name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {cfg.name}")
    if not cfg.api_key:
        raise ValueError(f"API key for {cfg.name} is empty")
    return PROVIDERS[cfg.name]["fn"](cfg, frames, user_ctx=user_ctx)
