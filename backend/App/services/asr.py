import os
import torch
import librosa

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoProcessor,
    AutoModelForSpeechSeq2Seq
)
from pdf import generate_pdf

# =========================
# DEVICE CONFIG
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

HF_TOKEN = os.getenv("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")  # set via environment variable


# =========================
# LOAD MODELS (LAZY LOAD)
# =========================

_whisper_model = None
_whisper_processor = None

_bangla_model = None
_bangla_processor = None


def load_whisper():
    global _whisper_model, _whisper_processor

    if _whisper_model is None:
        model_name = "openai/whisper-large-v3-turbo"

        _whisper_processor = WhisperProcessor.from_pretrained(
            model_name,
            token=HF_TOKEN
        )

        _whisper_model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
            token=HF_TOKEN
        ).to(DEVICE)

    return _whisper_processor, _whisper_model


def load_bangla():
    global _bangla_model, _bangla_processor

    if _bangla_model is None:
        model_name = "bangla-speech-processing/BanglaASR"

        _bangla_processor = AutoProcessor.from_pretrained(
            model_name,
            token=HF_TOKEN
            
        )

        _bangla_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
            token=HF_TOKEN
        ).to(DEVICE)

    return _bangla_processor, _bangla_model


# =========================
# AUDIO LOADER
# =========================

def load_audio(path, sr=16000):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


# =========================
# MAIN TRANSCRIBE FUNCTION
# =========================

def transcribe(audio_path: str, model: str = "whisper_turbo"):
    audio = load_audio(audio_path)

    if model == "whisper_turbo":
        return transcribe_whisper(audio)

    elif model == "bangla_asr":
        return transcribe_bangla(audio)

    else:
        raise ValueError(f"Unsupported ASR model: {model}")


# =========================
# WHISPER
# =========================

def transcribe_whisper(audio):
    processor, model = load_whisper()

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(DEVICE)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    text = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    return {"text": text}


# =========================
# BANGLA ASR
# =========================

def transcribe_bangla(audio):
    processor, model = load_bangla()

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(input_features)

    transcription = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return {"text": transcription}

def save_transcript(text, audio_path):
    os.makedirs("transcripts", exist_ok=True)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    transcript_path = f"transcripts/{base_name}_transcript.txt"

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(text)

    return transcript_path


if __name__ == "__main__":
    audio_file = "test3.wav"

    result = transcribe(audio_file, model="whisper_turbo")
    text = result["text"]

    transcript_path = save_transcript(text, audio_file)
    pdf_path = generate_pdf(text, audio_file)

    print("Transcript saved:", transcript_path)
    print("PDF saved:", pdf_path)