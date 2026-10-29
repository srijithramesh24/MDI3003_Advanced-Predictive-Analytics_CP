from __future__ import annotations

import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlunparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_timestamped_text(text: str) -> list[dict]:
    """Parse lines like [MM:SS] text, [HH:MM:SS] text, or 00:00:10 --> 00:00:15 text."""
    lines = text.replace("\r", "").splitlines()
    out = []
    pattern = re.compile(r"^\s*\[?(\d{1,2}:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?\]?\s*(?:-->\s*(\d{1,2}:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?)?\s*(.*)$")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if not match:
            continue
        # Split timestamp prefix robustly; parse optional end timestamp.
        arrow = "-->" in line
        if arrow:
            parts = re.split(r"\s*-->\s*", line, maxsplit=1)
            start_str = re.match(r"^\[?([\d:.,]+)\]?", parts[0]).group(1)
            right = re.match(r"^\[?([\d:.,]+)\]?\s*(.*)$", parts[1])
            if not right:
                continue
            end_str, body = right.group(1), right.group(2)
            start = timestamp_to_seconds(start_str)
            end = timestamp_to_seconds(end_str)
        else:
            m = re.match(r"^\[?([\d:.,]+)\]?\s*(?:[-–|]\s*)?(.*)$", line)
            if not m:
                continue
            start = timestamp_to_seconds(m.group(1))
            body = m.group(2)
            end = start + 10.0
        body = re.sub(r"<[^>]+>", " ", body)
        body = re.sub(r"\s+", " ", body).strip()
        if body and end >= start:
            out.append({"start": float(start), "end": float(end), "text": body})
    # Lines with a start timestamp only get an end equal to the next start when possible.
    for i in range(len(out)-1):
        if out[i]["end"] <= out[i]["start"] + 10.0:
            gap = out[i+1]["start"] - out[i]["start"]
            if 0 < gap < 60:
                out[i]["end"] = out[i+1]["start"]
    return out


def timestamp_to_seconds(value: str) -> float:
    value = value.strip("[] ").replace(",", ".")
    parts = value.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Invalid timestamp: {value}")


def parse_subtitles(text: str) -> list[dict]:
    """Parse standard SRT/VTT blocks with timestamps and HTML cue tags."""
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", text.strip())
    out = []
    time_re = re.compile(r"(?P<s>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3})\s*-->\s*(?P<e>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3})")
    for block in blocks:
        lines = [x.strip() for x in block.splitlines() if x.strip()]
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line and time_re.search(line)), None)
        if timing_index is None:
            continue
        m = time_re.search(lines[timing_index])
        try:
            start = timestamp_to_seconds(m.group("s"))
            end = timestamp_to_seconds(m.group("e"))
        except (ValueError, AttributeError):
            continue
        body_lines = lines[timing_index+1:]
        body = " ".join(body_lines)
        body = re.sub(r"<[^>]+>", " ", body)
        body = re.sub(r"\s+", " ", body).strip()
        if body and end >= start:
            out.append({"start": float(start), "end": float(end), "text": body})
    return out


def retrieve_evidence(question: str, segments: list[dict], top_k: int = 3) -> list[dict]:
    if not question.strip() or not segments:
        return []
    docs = [str(x.get("text", "")) for x in segments]
    if not any(docs):
        return []
    # Word + character n-grams improve retrieval for terminology and partial phrases.
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        matrix = vectorizer.fit_transform(docs + [question])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    except ValueError:
        scores = np.zeros(len(docs))
    order = np.argsort(scores)[::-1][:min(top_k, len(docs))]
    return [{
        "start": float(segments[i]["start"]),
        "end": float(segments[i]["end"]),
        "text": docs[i],
        "score": float(scores[i]),
    } for i in order]


def answer_from_evidence(question: str, evidence: list[dict]) -> tuple[str, str]:
    if not evidence:
        return ("I can't answer this reliably from the available lecture transcript. No sufficiently relevant evidence was retrieved. Try rephrasing the question or checking the transcript.", "abstain")
    top = evidence[0]
    if top["score"] < 0.12:
        return ("I don't have enough matching evidence in this lecture to answer confidently. This is an abstention, not proof that the topic is absent.", "abstain")
    # Extractive, evidence-grounded response: does not invent a generative explanation.
    chosen = [x for x in evidence if x["score"] >= max(0.12, top["score"] * 0.45)][:3]
    lines = [f"Based on the retrieved lecture evidence: {x['text']} **[{format_time(x['start'])}–{format_time(x['end'])}]**" for x in chosen]
    return ("\n\n".join(lines), "supported")


def _patch_pyav_metadata_errors():
    """
    Compatibility shim: faster-whisper versions may call av.open(..., metadata_errors=...),
    while newer PyAV builds removed that keyword. Remove only that keyword and preserve others.
    """
    try:
        import av
        if getattr(av, "_evidenceqa_open_patched", False):
            return
        original_open = av.open
        def compatible_open(*args, **kwargs):
            kwargs.pop("metadata_errors", None)
            return original_open(*args, **kwargs)
        av.open = compatible_open
        av._evidenceqa_open_patched = True
    except Exception:
        # If PyAV isn't importable, the later model import will produce an actionable error.
        return


def _load_whisper_model(model_size: str):
    _patch_pyav_metadata_errors()
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def _language_code(language_choice: str):
    mapping = {
        "English": "en", "Hindi": "hi", "Malayalam": "ml", "Tamil": "ta",
        "Spanish": "es", "French": "fr", "German": "de",
    }
    return mapping.get(language_choice)


def _transcribe_path(path: str, model_size: str, language_choice: str):
    try:
        import streamlit as st
        model = _cached_model(model_size)
    except Exception as exc:
        raise RuntimeError(
            "Could not load the speech-recognition model. Check internet access for the first model download "
            "and install requirements in the same Python environment that starts Streamlit. "
            f"Details: {type(exc).__name__}: {exc}"
        ) from exc
    language = _language_code(language_choice)
    try:
        iterator, info = model.transcribe(
            path,
            language=language,
            vad_filter=True,
            beam_size=5,
            condition_on_previous_text=False,
        )
        rows = []
        for seg in iterator:
            body = (seg.text or "").strip()
            if body:
                rows.append({"start": float(seg.start), "end": float(seg.end), "text": body})
    except Exception as exc:
        raise RuntimeError(
            "Audio decoding/transcription failed. Try MP4 (H.264/AAC), MP3/WAV, or upload subtitles. "
            f"Details: {type(exc).__name__}: {exc}"
        ) from exc
    if not rows:
        raise RuntimeError("No speech was detected. Check audio volume/language or upload subtitles.")
    return rows, getattr(info, "language", None)


def _cached_model(model_size: str):
    # Streamlit cache avoids reloading the speech model on every interaction.
    import streamlit as st
    return _cached_model_impl(model_size)


# Avoid requiring Streamlit cache in unit tests or non-Streamlit callers.
_MODEL_CACHE = {}
def _cached_model_impl(model_size: str):
    if model_size not in _MODEL_CACHE:
        _MODEL_CACHE[model_size] = _load_whisper_model(model_size)
    return _MODEL_CACHE[model_size]


def transcribe_local_media(media_bytes: bytes, filename: str, model_size: str, language_choice: str):
    if not media_bytes:
        raise ValueError("The uploaded file is empty.")
    suffix = Path(filename).suffix.lower()
    allowed = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
    if suffix not in allowed:
        raise ValueError(f"Unsupported file extension {suffix}. Choose a supported audio/video file.")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(media_bytes)
            temp_path = temp.name
        return _transcribe_path(temp_path, model_size, language_choice)
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass


def canonical_youtube_url(url: str) -> str:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]
    elif host.endswith("youtube.com") or host == "youtube.com":
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/embed/", "/shorts/", "/live/")):
            video_id = parsed.path.split("/")[2] if len(parsed.path.split("/")) > 2 else ""
        else:
            video_id = ""
    else:
        raise ValueError("Please enter a valid YouTube URL (youtube.com or youtu.be).")
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id or ""):
        raise ValueError("Could not identify a valid 11-character YouTube video ID from this URL.")
    return f"https://www.youtube.com/watch?v={video_id}"


def download_youtube_media(url: str, model_size: str, language_choice: str):
    canonical = canonical_youtube_url(url)
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is not installed. Run: python -m pip install -r requirements.txt") from exc

    temp_dir = tempfile.TemporaryDirectory(prefix="evidenceqa_yt_")
    folder = Path(temp_dir.name)
    output_template = str(folder / "lecture.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 2,
        "fragment_retries": 2,
        "socket_timeout": 25,
        "restrictfilenames": True,
    }
    try:
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(canonical, download=True)
                title = info.get("title", "YouTube lecture")
        except Exception as exc:
            raise RuntimeError(
                "Could not download this YouTube video's audio. It may be private, age/region restricted, "
                "unavailable, blocked by the network, or require sign-in. Use an accessible lecture or upload subtitles. "
                f"Details: {type(exc).__name__}: {exc}"
            ) from exc

        candidates = [p for p in folder.iterdir() if p.is_file() and p.stat().st_size > 0]
        if not candidates:
            raise RuntimeError("YouTube downloader returned no audio file.")
        # Pick largest file; yt-dlp may add a format suffix.
        audio_path = max(candidates, key=lambda p: p.stat().st_size)
        rows, detected = _transcribe_path(str(audio_path), model_size, language_choice)
        return rows, detected, canonical, title
    finally:
        temp_dir.cleanup()
