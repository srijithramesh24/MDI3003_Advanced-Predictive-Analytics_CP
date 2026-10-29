# EvidenceQA
## Evidence-Grounded Lecture Video Question Answering with Timestamp-Level Support

EvidenceQA is a research prototype that lets a learner ask questions about a lecture and inspect the transcript evidence supporting the response. It indexes lecture speech into timestamped segments, retrieves relevant passages for a question, and presents the supporting text with timestamps so the learner can verify the answer against the original lecture.

The project is designed to demonstrate **evidence-grounded question answering**, not to act as an unrestricted general-purpose chatbot.

## Problem Statement

Students often need to locate a specific explanation inside a long lecture recording. Watching or scrubbing through the entire video is time-consuming, while ordinary chat systems may provide answers without showing where the information came from.

EvidenceQA explores a workflow in which answers are tied to retrieved lecture-transcript passages and their timestamps. When the available transcript does not provide sufficiently relevant evidence, the system should abstain rather than confidently invent an answer.

## Objectives

- Accept lecture media, a supported YouTube URL, or a transcript/subtitle file or pasted timestamped transcript.
- Transcribe speech into text segments with timestamps when audio transcription is available.
- Retrieve transcript passages relevant to a learner's question.
- Display answers with source excerpts and timestamp references.
- Provide links to the corresponding YouTube timestamp when supported by the source URL.
- Abstain when retrieved evidence does not meet the configured relevance threshold.
- Let users inspect the transcript and export retrieved evidence where implemented.

## Main Features

### 1. Multiple lecture input methods
- Upload a local video or audio file.
- Provide a YouTube lecture URL for extraction, subject to video accessibility and platform restrictions.
- Upload transcript/subtitle files.
- Paste a timestamped transcript.
- Use the built-in demo workflow.

### 2. Timestamped transcription
The application can use `faster-whisper` to transcribe supported local media. The selected model controls the speed/accuracy trade-off:

- `tiny`: fastest and suitable for a quick smoke test; accuracy may be lower.
- `base` or larger models: may improve transcription quality but require more time, memory, and model downloads.

The first transcription run may take longer because model files need to be downloaded. Transcription time depends on the recording length, CPU/GPU, model size, audio quality, and available memory.

### 3. Evidence retrieval and answers
The application divides a transcript into timestamped segments and retrieves passages relevant to the submitted question. The interface displays the retrieved passages, their timestamps, and retrieval relevance scores. These scores are retrieval signals; they are **not calibrated probabilities that an answer is correct**.

### 4. Evidence-based abstention
If no retrieved passage passes the configured relevance threshold, the application should indicate that it cannot answer reliably from the available lecture evidence. Users can then refine the question or inspect the transcript.

### 5. YouTube timestamp references
When a supported YouTube URL is supplied and the video can be accessed, evidence passages may include links that open the video at the relevant timestamp. Extraction can fail for private, age-restricted, region-blocked, live, DRM-protected, or otherwise inaccessible videos. Network availability and YouTube-side changes can also affect extraction.

## Technology Stack

- **Language:** Python
- **Web interface:** Streamlit
- **Speech recognition:** `faster-whisper`
- **Transcript processing and retrieval:** Python text-processing and relevance-scoring components in `src/`
- **Testing:** pytest

Check `requirements.txt` for the project's installed Python dependencies.

## Project Structure

```text
EvidenceQA/
├── app.py                 # Streamlit application entry point
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── run_windows.bat        # Windows launcher, if included in this copy
├── src/                   # Transcription, indexing, retrieval, and helper modules
├── tests/                 # Automated tests
├── streamlit/             # Streamlit configuration, if included
└── Prototype_Screenshots/ # Optional screenshots of the running prototype
```

The exact files may vary slightly by project version. Keep the `src/` and other project folders supplied with the code; do not copy only `app.py` into a new folder.

## Requirements

- Windows 10/11 or another OS supported by Python and Streamlit.
- Python **3.11 or 3.12 recommended** for this project version.
- Internet access for installing packages and downloading the speech-recognition model on first use.
- Sufficient disk space for Python packages and model files.
- A supported local media file, transcript, or accessible YouTube lecture URL.

Python 3.14 may not be compatible with every dependency in the current environment. If installation or imports fail on Python 3.14, install Python 3.11 or 3.12 and create a fresh virtual environment using that interpreter.

## Installation on Windows

Open **Command Prompt** (not the Ubuntu/WSL terminal) and change to the project directory. Replace the example path with the actual folder where you extracted EvidenceQA.

```bat
cd /d "C:\Users\YOUR_USERNAME\Downloads\EvidenceQA_ReviewReady_Stable\EvidenceQA_ReviewReady_Stable"
```

Confirm that the project files are present:

```bat
dir
```

Create a virtual environment:

```bat
py -3.12 -m venv .venv
```

Activate it:

```bat
.venv\Scripts\activate
```

Upgrade pip and install dependencies:

```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the web application:

```bat
python -m streamlit run app.py
```

Open the local URL printed in the terminal, normally:

```text
http://localhost:8501
```

**If `py -3.12` is not recognized:** install Python 3.12 from the official Python website and select the option to add Python to PATH. Then reopen Command Prompt and retry. If the project already has a working `.venv`, do not recreate it while the application is running.

**If the virtual environment is missing or broken:** from the project root, run:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If the environment was created with a different Python version, delete `.venv` only after stopping the app, then recreate it with Python 3.11 or 3.12.

## How to Use

1. Start EvidenceQA and open the local web page.
2. Select an input method in the sidebar.
3. For local media, upload a video/audio file, select a speech-recognition model and language, then run transcription.
4. For a YouTube source, enter a supported lecture URL and start extraction/transcription.
5. Alternatively, upload subtitles/transcripts, paste a timestamped transcript, or use the demo.
6. Wait until the application reports that the transcript has been indexed.
7. Enter a question that the lecture is expected to answer.
8. Select the number of evidence passages and minimum relevance threshold.
9. Choose **Find evidence and answer**.
10. Read the answer, inspect each evidence passage, and verify its timestamp against the original lecture.

### Example test questions

Use questions that are explicitly answered in the selected lecture. For a lecture about SQL/DBMS, examples could include:

- “What does SQL stand for?”
- “What are the main categories of SQL commands?”
- “What is the difference between DDL and DML?”
- “What are the ACID properties?”

The example questions above are only suitable if the selected lecture actually covers those topics. A question about a topic absent from the transcript should result in an abstention.

## Troubleshooting

### Transcription appears to run for several minutes
The first run may download the selected Whisper model. Longer recordings and larger models take more time. Start with a short, clear lecture and the `tiny` model to verify the workflow before trying a long recording.

### Missing `faster_whisper`
Activate the project's virtual environment and install the dependencies again:

```bat
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

### Media upload succeeds, but audio decoding fails
Try playing the file locally first. If necessary, export it as MP4 with H.264 video and AAC audio, or convert the audio to WAV/MP3. You can also use a subtitle/transcript input as an alternative. Media container support depends on the installed decoding libraries.

### YouTube extraction fails
Confirm that the URL is valid and that the video is publicly accessible from your network. Try a different permitted lecture URL or provide an SRT/VTT transcript. The application cannot guarantee extraction from every YouTube video.

### `streamlit run .\app.py` says the target is not a Python file
Run the command from the project root and ensure the filename is exactly `app.py`:

```bat
dir app.py
python -m streamlit run app.py
```

### `The system cannot find the path specified`
The command was run from the wrong folder or the virtual environment does not exist. First navigate to the directory that contains `app.py` and `requirements.txt`, then create/activate `.venv` as described above.

## Research Scope and Limitations

EvidenceQA is a prototype intended for demonstration and research evaluation. Its output quality depends on the transcript quality, timestamp accuracy, retrieval method, and the relevance threshold. Automatic speech recognition can misrecognize technical terms, names, accents, or noisy speech. Retrieval relevance is not a guarantee of factual correctness, and an answer assembled from transcript excerpts can still be incomplete or misleading.

YouTube access is not guaranteed for every URL. Timestamp links identify a location in the source video; they do not independently prove that the transcript is correct.

For a formal research evaluation, test on a documented set of lecture videos and questions, manually verify evidence spans and timestamps, report failure cases, and compare the system against defined baselines. Do not present demo data or a few successful examples as proof of general accuracy. Obtain permission before processing or redistributing copyrighted lecture recordings.

## Testing

Run the project's tests from the root directory with the virtual environment activated:

```bat
python -m pytest -q
```

A passing test suite checks only the behaviors covered by those tests; it does not establish research validity or guarantee correct answers for all lectures.

## Screenshots

If the `Prototype_Screenshots/` directory is included, it contains images of the interface for documentation. Screenshots illustrate the prototype UI and should not be treated as experimental results unless the corresponding behavior has been tested and verified.

## Responsible Use

- Use lecture material you are permitted to access and process.
- Verify answers against the cited transcript and original video before relying on them.
- Avoid uploading confidential or personally identifiable recordings without authorization.
- Clearly distinguish prototype demonstrations from measured research results.

## Project Title

**Evidence-Grounded Lecture Video Question Answering with Timestamp-Level Support**

## License

No license is specified in this repository. Add an appropriate license file before allowing others to reuse, modify, or redistribute the project.
