# EvidenceQA — Stable Review Prototype

**Title:** Evidence-Grounded Lecture Video Question Answering with Timestamp-Level Support

## What it does
- Upload a local video/audio file and transcribe it into timestamped segments using faster-whisper.
- Accept YouTube URLs via `yt-dlp` for videos accessible without bypassing sign-in, age, region, or DRM restrictions.
- Accept SRT/VTT subtitle files or pasted timestamped transcripts.
- Retrieve transcript evidence using TF-IDF and cosine similarity.
- Return extractive, evidence-grounded responses with timestamps and abstain when retrieval evidence is insufficient.
- Export retrieved evidence as CSV.

## Windows setup (recommended Python 3.11 or 3.12)
Open Command Prompt in the extracted folder:

```cmd
py -0p
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

If `py -3.12` fails, install Python 3.12 from https://www.python.org/downloads/release/python-31210/ and retry. The first Whisper run needs internet to download the model. Use the same `.venv` Python command for installation and launching.

## Fix for `metadata_errors`
The app includes a runtime compatibility shim that removes the `metadata_errors` keyword when the installed PyAV no longer accepts it. This addresses the specific error seen with some `faster-whisper`/PyAV version combinations. The requirements also request a compatible PyAV baseline. If the environment is already inconsistent, recreate it using the commands above.

## YouTube URL support
Supported URL forms include `youtube.com/watch?v=VIDEO_ID`, `youtu.be/VIDEO_ID`, `/shorts/VIDEO_ID`, `/embed/VIDEO_ID`, and `/live/VIDEO_ID`. The app extracts audio with `yt-dlp`, then transcribes it. URL downloading is not guaranteed for private, age-restricted, region-blocked, sign-in-required, live, removed, or otherwise inaccessible videos. Do not use this tool to bypass access controls. The network must allow YouTube access.

## Research integrity and limitations
- This is a research prototype, not a validated production system.
- Transcript retrieval is TF-IDF/cosine similarity; its score is a ranking score, not calibrated confidence.
- The answer is extractive and displays source text; it does not claim to generate a fully reasoned LLM answer.
- Manually verify timestamps and answers against the source video.
- Use a labelled evaluation set and compare against baselines before making performance or novelty claims.
- Demo data is only for testing the interface, not experimental evidence.

## Run tests
```cmd
.venv\Scripts\python.exe -m pytest -q
```
