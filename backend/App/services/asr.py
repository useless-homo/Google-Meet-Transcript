from google.cloud import speech

def transcribe(audio_path):
    client = speech.SpeechClient()

    with open(audio_path, "rb") as f:
        content = f.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        language_code="en-US",
        enable_word_time_offsets=True,
        enable_speaker_diarization=True,
        diarization_speaker_count=2,
    )

    response = client.recognize(config=config, audio=audio)

    entries = []
    for result in response.results:
        alt = result.alternatives[0]
        entries.append({
            "text": alt.transcript,
            "confidence": alt.confidence
        })

    return entries
