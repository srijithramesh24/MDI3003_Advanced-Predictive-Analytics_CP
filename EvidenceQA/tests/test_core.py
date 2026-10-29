from src.core import parse_subtitles, parse_timestamped_text, canonical_youtube_url, retrieve_evidence, answer_from_evidence

def test_parse_timestamped_text():
    rows = parse_timestamped_text("[00:00] Intro\n[00:15] Overfitting happens when a model memorizes noise")
    assert len(rows) == 2
    assert rows[1]["start"] == 15
    assert "Overfitting" in rows[1]["text"]

def test_parse_srt():
    srt = "1\n00:00:01,000 --> 00:00:03,500\nHello lecture\n\n2\n00:00:03,500 --> 00:00:05,000\nSecond line"
    rows = parse_subtitles(srt)
    assert len(rows) == 2
    assert rows[0]["start"] == 1
    assert rows[0]["end"] == 3.5

def test_youtube_urls():
    assert canonical_youtube_url("https://youtu.be/abcdefghijk?t=3") == "https://www.youtube.com/watch?v=abcdefghijk"
    assert canonical_youtube_url("https://www.youtube.com/watch?v=abcdefghijk") == "https://www.youtube.com/watch?v=abcdefghijk"

def test_retrieval_and_abstention():
    rows = [{"start": 0, "end": 10, "text": "Overfitting is when a model memorizes training data."},
            {"start": 10, "end": 20, "text": "Precision measures predicted positives that are correct."}]
    results = retrieve_evidence("What is overfitting?", rows, top_k=2)
    assert results and "Overfitting" in results[0]["text"]
    answer, status = answer_from_evidence("What is overfitting?", results)
    assert status == "supported"
    answer, status = answer_from_evidence("What is quantum entanglement?", [])
    assert status == "abstain"
