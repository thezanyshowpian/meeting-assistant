"""
Interactive interface.

Deliberate UI choices, each tied to something the PS scores:
  - raw and refined transcripts shown SIDE BY SIDE, so the effect of Stage 2 is
    visible rather than asserted;
  - applied corrections shown with their reasons, and REJECTED ones shown too —
    the rejections are evidence that the verification gates are doing work;
  - missing owners/deadlines rendered explicitly as "unspecified", never blank,
    so the viewer can see we didn't guess;
  - clear processing status and clear failure messages.

Run:  streamlit run app/ui/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.export import to_json, to_markdown, transcripts_markdown  # noqa: E402
from app.pipeline.glossary import Glossary, default_glossary  # noqa: E402
from app.pipeline.orchestrator import run_pipeline  # noqa: E402
from app.schemas import UNSPECIFIED  # noqa: E402

st.set_page_config(page_title="Meeting Assistant", page_icon="📝", layout="wide")

st.title("AI Meeting Assistant")
st.caption("Transcribe → correct domain terminology → generate minutes, "
           "decisions and action items.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    uploaded_glossary = st.file_uploader(
        "Domain glossary (JSON)", type=["json"],
        help='A list of your domain terms, e.g. ["Kubernetes", "PostgreSQL"]. '
             "Used to correct mis-heard jargon.",
    )
    use_llm = st.checkbox(
        "Use the language model for refinement", value=True,
        help="Off = deterministic phonetic glossary matching only (no LLM).",
    )
    st.divider()
    glossary = default_glossary()
    if uploaded_glossary is not None:
        try:
            import json
            data = json.loads(uploaded_glossary.read().decode())
            glossary = Glossary(data["terms"] if isinstance(data, dict) else data)
            st.success(f"Loaded {len(glossary)} custom terms.")
        except Exception as exc:                                   # noqa: BLE001
            st.error(f"Could not read that glossary file: {exc}")
    else:
        st.info(f"Using the built-in sample glossary ({len(glossary)} terms).")

# ------------------------------------------------------------------- upload
audio_file = st.file_uploader(
    "Upload a meeting recording", type=["wav", "mp3", "m4a", "flac", "ogg", "mp4", "webm"],
)

if audio_file and st.button("Process recording", type="primary"):
    suffix = os.path.splitext(audio_file.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_file.getbuffer())
        tmp_path = tmp.name

    status = st.status("Starting…", expanded=True)

    def progress(stage: str, message: str) -> None:
        status.write(f"**{stage}** — {message}")

    try:
        result = run_pipeline(tmp_path, glossary=glossary,
                              use_llm_refinement=use_llm, progress=progress)
        status.update(label="Done", state="complete", expanded=False)
        st.session_state["result"] = result
    except Exception as exc:                                       # noqa: BLE001
        status.update(label="Failed", state="error")
        st.error(f"Processing failed: {exc}")
    finally:
        os.unlink(tmp_path)

# ------------------------------------------------------------------ results
result = st.session_state.get("result")
if result:
    if not result.ok:
        st.error(f"**{result.failed_stage} failed:** {result.error}")
    for warning in result.warnings:
        st.warning(warning)

    if result.stage_times:
        cols = st.columns(len(result.stage_times))
        for col, (stage, seconds) in zip(cols, result.stage_times.items()):
            col.metric(stage, f"{seconds:.1f}s")

    tabs = st.tabs(["Transcripts", "Meeting record", "Speakers", "Corrections",
                    "Downloads"])

    # -- transcripts, side by side so Stage 2's effect is visible
    with tabs[0]:
        left, right = st.columns(2)
        with left:
            st.subheader("Raw transcript")
            st.caption("Straight from speech-to-text, before any correction.")
            st.text_area("raw", result.raw_text, height=400, label_visibility="collapsed")
        with right:
            st.subheader("Refined transcript")
            st.caption("After domain-terminology correction.")
            st.text_area("refined", result.refined_text, height=400,
                         label_visibility="collapsed")

    # -- the meeting record
    with tabs[1]:
        record = result.record
        if not record:
            st.info("No meeting record was produced.")
        else:
            if record.summary:
                st.subheader("Summary")
                st.write(record.summary)

            st.subheader("Minutes")
            if record.minutes:
                for item in record.minutes:
                    st.markdown(f"- {item}")
            else:
                st.caption("No minutes recorded.")

            st.subheader("Key decisions")
            if record.decisions:
                for d in record.decisions:
                    st.markdown(f"**{d.statement}**")
                    if d.evidence:
                        st.caption(f'evidence: "{d.evidence}"')
            else:
                st.caption("No decisions were reached in this meeting.")

            st.subheader("Action items")
            if record.action_items:
                st.table([
                    {"Task": a.task,
                     "Owner": a.owner_display,
                     "Deadline": a.deadline or UNSPECIFIED,
                     "How the owner is known": a.owner_evidence or "—"}
                    for a in record.action_items
                ])
                st.caption(f'"{UNSPECIFIED}" means the recording did not state it. '
                           "We never guess an owner or a deadline. *(voice only)*: "
                           "that voice committed in the first person; *(inferred)*: "
                           "that voice was also linked to a name by evidence in the "
                           "conversation (see the Speakers tab).")
            else:
                st.caption("No action items were assigned.")

    # -- who spoke, and which names the conversation actually supports
    with tabs[2]:
        if not result.diarization:
            st.info("Speaker diarization did not run (install "
                    "requirements-diarization.txt, or check DIARIZE).")
        else:
            d, naming = result.diarization, result.naming
            st.caption(f"{d.num_speakers} speakers — {d.method}")
            st.subheader("Names inferred from the conversation")
            if naming and naming.bindings:
                st.table([{"Speaker": b.speaker, "Name (inferred)": b.name,
                           "Evidence": "; ".join(e.describe() for e in b.evidence)}
                          for b in naming.bindings.values()])
            unnamed = [f"Speaker {i}" for i in range(1, d.num_speakers + 1)
                       if not (naming and naming.name_for(f"Speaker {i}"))]
            if unnamed:
                st.caption(f"No evidence for a name: {', '.join(unnamed)} — left "
                           "anonymous rather than guessed.")
            for c in (naming.conflicts if naming else []):
                st.caption(f"Not named: {c}")
            st.subheader("Speaker-labelled transcript")
            show = naming.display if naming else (lambda x: x)
            for u in (result.utterances or []):
                st.markdown(f"**{show(u.speaker)}** `[{u.start:.0f}s]` {u.text}")

    # -- corrections, including the rejected ones
    with tabs[3]:
        ref = result.refinement
        if not ref:
            st.info("Refinement did not run.")
        else:
            st.subheader(f"Applied ({len(ref.applied)})")
            if ref.applied:
                st.table([{"Original": e.original, "Corrected": e.replacement,
                           "Type": e.edit_type.value, "Reason": e.reason}
                          for e in ref.applied])
            else:
                st.caption("No corrections were needed.")

            st.subheader(f"Rejected by verification ({len(ref.rejected)})")
            st.caption("Proposed changes that failed the safety gates. Shown so you "
                       "can see what the system refused to do.")
            if ref.rejected:
                st.table([{"Original": e.original, "Proposed": e.replacement,
                           "Rejected because": "; ".join(e.rejections)}
                          for e in ref.rejected])
            else:
                st.caption("Nothing was rejected.")

    # -- downloads: same object, two formats, so they cannot disagree
    with tabs[4]:
        st.download_button("Meeting record (JSON — machine-readable)",
                           to_json(result), "meeting_record.json", "application/json")
        st.download_button("Meeting record (Markdown — human-readable)",
                           to_markdown(result), "meeting_record.md", "text/markdown")
        st.download_button("Transcripts (Markdown)",
                           transcripts_markdown(result), "transcripts.md", "text/markdown")
