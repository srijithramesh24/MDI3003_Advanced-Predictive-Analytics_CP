from __future__ import annotations

import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import pandas as pd
import streamlit as st

from src.core import (
    parse_subtitles,
    parse_timestamped_text,
    retrieve_evidence,
    answer_from_evidence,
    format_time,
    transcribe_local_media,
    download_youtube_media,
)

st.set_page_config(page_title="EvidenceQA", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1500px;}
.hero {padding: 1.4rem 1.6rem; border: 1px solid rgba(128,128,128,.28); border-radius: 16px; margin-bottom: 1rem;}
.small-muted {color: #9ca3af; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

st.title("🎓 EvidenceQA")
st.caption("Evidence-Grounded Lecture Video Question Answering with Timestamp-Level Support")
st.markdown(
    "Ask a lecture a question, inspect the exact transcript evidence, and jump to the supporting timestamp. "
    "Answers are limited to retrieved transcript evidence; this prototype is not a general-purpose chatbot."
)

for key, default in {
    "segments": [],
    "source_label": "",
    "media_url": "",
    "video_path": "",
    "last_transcription_error": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("1. Add a lecture")
    method = st.radio(
        "Input method",
        ["Upload video/audio", "YouTube URL", "Upload transcript/subtitles", "Paste timestamped transcript", "Try demo"],
        help="YouTube extraction uses yt-dlp and works only for videos accessible to your account/region and not protected by DRM."
    )
    st.divider()
    st.subheader("Transcription settings")
    model_size = st.selectbox("Speech-recognition model", ["tiny", "base", "small"], index=0)
    language = st.selectbox(
        "Lecture language",
        ["Auto-detect", "English", "Hindi", "Malayalam", "Tamil", "Spanish", "French", "German"],
        index=0
    )
    st.caption("tiny is fastest. base/small can improve accuracy but need more time and memory.")

def set_segments(segments, label):
    st.session_state.segments = segments
    st.session_state.source_label = label
    st.session_state.last_transcription_error = ""

if method == "Upload video/audio":
    upload = st.sidebar.file_uploader(
        "Choose a local media file",
        type=["mp4", "mov", "mkv", "webm", "avi", "m4v", "mp3", "wav", "m4a", "aac", "flac", "ogg"],
        help="Start with a 5–15 minute MP4 with clear speech."
    )
    if upload:
        raw = upload.getvalue()
        st.sidebar.success(f"Selected: {upload.name} ({len(raw)/(1024*1024):.1f} MB)")
        if Path(upload.name).suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
            st.video(raw)
        if st.sidebar.button("Transcribe media and index timestamps", type="primary", use_container_width=True):
            with st.spinner("Transcribing speech. First run may download the model…"):
                try:
                    segments, detected = transcribe_local_media(raw, upload.name, model_size, language)
                    set_segments(segments, upload.name)
                    st.session_state.media_url = ""
                    st.sidebar.success(f"Done: {len(segments)} timestamped segments. Language: {detected or 'unknown'}.")
                except Exception as exc:
                    st.session_state.last_transcription_error = str(exc)
                    st.sidebar.error("Transcription failed. See troubleshooting below.")
                    st.error(st.session_state.last_transcription_error)

elif method == "YouTube URL":
    url = st.sidebar.text_input("YouTube lecture URL", placeholder="https://www.youtube.com/watch?v=...")
    st.sidebar.caption("Public/unlisted links may work if accessible. Private, age-restricted, region-blocked, live, or DRM-protected videos may fail.")
    if st.sidebar.button("Extract audio and transcribe", type="primary", use_container_width=True):
        if not url.strip():
            st.sidebar.warning("Paste a YouTube URL first.")
        else:
            with st.spinner("Validating URL, extracting audio, and transcribing…"):
                try:
                    segments, detected, canonical_url, title = download_youtube_media(url.strip(), model_size, language)
                    set_segments(segments, title or canonical_url)
                    st.session_state.media_url = canonical_url
                    st.sidebar.success(f"Done: {len(segments)} timestamped segments. Language: {detected or 'unknown'}.")
                    st.info(f"Lecture loaded: **{title or canonical_url}**")
                except Exception as exc:
                    st.session_state.last_transcription_error = str(exc)
                    st.error(f"YouTube extraction failed: {exc}")
                    st.info("Fallback: download captions/subtitles you are permitted to access and use **Upload transcript/subtitles**.")

elif method == "Upload transcript/subtitles":
    transcript_file = st.sidebar.file_uploader(
        "Choose SRT, VTT, or TXT",
        type=["srt", "vtt", "txt", "csv"],
        key="transcript_upload"
    )
    if transcript_file:
        content = transcript_file.getvalue().decode("utf-8-sig", errors="replace")
        if st.sidebar.button("Load transcript", type="primary", use_container_width=True):
            try:
                suffix = Path(transcript_file.name).suffix.lower()
                if suffix in {".srt", ".vtt"}:
                    segments = parse_subtitles(content)
                else:
                    segments = parse_timestamped_text(content)
                if not segments:
                    raise ValueError("No timestamped segments found. SRT/VTT or [MM:SS] text is required.")
                set_segments(segments, transcript_file.name)
                st.sidebar.success(f"Loaded {len(segments)} transcript segments.")
            except Exception as exc:
                st.sidebar.error(str(exc))

elif method == "Paste timestamped transcript":
    transcript_text = st.sidebar.text_area(
        "Paste transcript",
        height=220,
        placeholder="[00:00] Welcome to this lecture.\n[00:14] Overfitting occurs when..."
    )
    if st.sidebar.button("Index pasted transcript", type="primary", use_container_width=True):
        try:
            segments = parse_timestamped_text(transcript_text)
            if not segments:
                raise ValueError("No timestamps detected. Use [MM:SS] or [HH:MM:SS] at the start of each line.")
            set_segments(segments, "Pasted transcript")
            st.sidebar.success(f"Indexed {len(segments)} segments.")
        except Exception as exc:
            st.sidebar.error(str(exc))

else:
    if st.sidebar.button("Load demo lecture", type="primary", use_container_width=True):
        demo = [
            (0, 14, "Welcome. Today we will discuss machine learning model evaluation and overfitting."),
            (14, 32, "Overfitting happens when a model learns the training examples too closely, including noise, and performs poorly on unseen data."),
            (32, 49, "A common symptom is high training accuracy combined with much lower validation accuracy."),
            (49, 68, "To reduce overfitting, use cross-validation, regularization, simpler models, and more representative training data."),
            (68, 84, "A decision tree can overfit when it grows too deep and creates leaves for very small groups of examples."),
            (84, 101, "Pruning limits tree complexity by removing branches that do not improve generalization."),
            (101, 120, "The validation set is used to compare model choices. It should not be treated as the final unbiased test result."),
            (120, 137, "The test set is reserved for a final evaluation after model selection is complete."),
            (137, 154, "Precision measures the fraction of predicted positives that are actually positive. Recall measures the fraction of actual positives that are found."),
            (154, 172, "F1 is the harmonic mean of precision and recall. It is useful when both false positives and false negatives matter."),
        ]
        set_segments([{"start": a, "end": b, "text": t} for a,b,t in demo], "Demo: Model Evaluation")
        st.session_state.media_url = ""
        st.sidebar.success("Demo lecture loaded.")

st.divider()
left, right = st.columns([1.35, 1])
with left:
    st.header("2. Ask the lecture")
    if not st.session_state.segments:
        st.info("Add a video, YouTube URL, timestamped transcript, or demo lecture from the sidebar to begin.")
    else:
        st.success(f"Evidence index ready: **{st.session_state.source_label}** · {len(st.session_state.segments)} segments")
        question = st.text_input("Question", placeholder="What is overfitting, and how can it be reduced?")
        top_k = st.slider("Evidence passages to retrieve", min_value=1, max_value=5, value=3)
        threshold = st.slider("Minimum evidence relevance", min_value=0.05, max_value=0.60, value=0.12, step=0.01)
        if st.button("Find evidence and answer", type="primary", disabled=not bool(question.strip())):
            results = retrieve_evidence(question, st.session_state.segments, top_k=top_k)
            results = [r for r in results if r["score"] >= threshold]
            answer, status = answer_from_evidence(question, results)
            st.session_state.last_question = question
            st.session_state.last_results = results
            st.session_state.last_answer = answer
            st.session_state.last_status = status

        if "last_answer" in st.session_state and st.session_state.get("last_question"):
            st.subheader("Answer")
            if st.session_state.last_status == "supported":
                st.markdown(st.session_state.last_answer)
                st.caption("Answer generated from retrieved transcript excerpts; verify against the lecture.")
            else:
                st.warning(st.session_state.last_answer)
            results = st.session_state.get("last_results", [])
            st.subheader("Evidence passages")
            if not results:
                st.info("No passage met the selected relevance threshold. Lower the threshold slightly or ask a more specific question.")
            for idx, item in enumerate(results, start=1):
                start, end = item["start"], item["end"]
                with st.expander(f"Evidence {idx} · {format_time(start)}–{format_time(end)} · relevance {item['score']:.3f}", expanded=(idx == 1)):
                    st.write(item["text"])
                    if st.session_state.media_url and ("youtube.com" in st.session_state.media_url or "youtu.be" in st.session_state.media_url):
                        sep = "&" if "?" in st.session_state.media_url else "?"
                        st.markdown(f"[Open YouTube at {format_time(start)}]({st.session_state.media_url}{sep}t={int(start)}s)")
                    st.caption("Relevance is a retrieval score, not a calibrated probability of correctness.")
            if results:
                evidence_df = pd.DataFrame([{
                    "rank": i, "start_seconds": r["start"], "end_seconds": r["end"],
                    "start_timestamp": format_time(r["start"]), "end_timestamp": format_time(r["end"]),
                    "relevance_score": round(r["score"], 4), "evidence_text": r["text"]
                } for i, r in enumerate(results, 1)])
                st.download_button("Download evidence as CSV", evidence_df.to_csv(index=False).encode("utf-8"), "evidenceqa_evidence.csv", "text/csv")

with right:
    st.header("3. Grounding & evaluation")
    st.markdown("""
    - Retrieve timestamped transcript passages.
    - Answer only when evidence passes the relevance threshold.
    - Otherwise abstain rather than fabricate an answer.
    - Display source excerpts and timestamps for manual verification.
    """)
    if st.session_state.segments:
        st.metric("Indexed segments", len(st.session_state.segments))
        st.metric("Total transcript duration", format_time(max(x["end"] for x in st.session_state.segments)))
        st.subheader("Transcript preview")
        for seg in st.session_state.segments[:12]:
            st.markdown(f"**{format_time(seg['start'])}–{format_time(seg['end'])}** — {seg['text']}")
        if len(st.session_state.segments) > 12:
            st.caption(f"Showing first 12 of {len(st.session_state.segments)} segments.")

st.divider()
with st.expander("Troubleshooting: upload, transcription, or YouTube extraction"):
    st.markdown("""
    **If the error mentions `metadata_errors`:** this is a PyAV/faster-whisper compatibility issue. The app includes a compatibility shim that removes that obsolete keyword before PyAV is called. Restart this app from the same environment after replacing the code.

    **If YouTube extraction fails:** install dependencies from this project's `requirements.txt`; make sure the URL is a normal video URL, the video is accessible without login/age/region restrictions, and your network permits the download. Some videos disable extraction. The app does not bypass DRM or access controls.

    **If transcription detects no speech:** confirm the video has clear audio, choose its language explicitly, or upload SRT/VTT captions.

    **For reliable research results:** manually verify timestamp correctness and evaluate against annotated questions. Demo data is for interface testing only and must not be reported as experimental results.
    """)

st.caption("Research prototype. Transcript retrieval scores are not calibrated confidence values; verify answers and timestamps against the source.")
