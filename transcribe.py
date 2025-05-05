# voice_agent/transcribe.py
from faster_whisper import WhisperModel
import os
import traceback

model_size = "base"
compute_type = "int8"

transcribe_model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
print(f"Faster Whisper model '{model_size}' loaded.")

def transcribe_audio(filepath="input.wav"):
    if transcribe_model is None: return "Error: Transcription model not loaded."
    if not os.path.exists(filepath): return f"Error: Audio file missing ({filepath})"

    print("🧠 Transcribing...")
    segments, info = transcribe_model.transcribe(filepath, beam_size=5, vad_filter=True)
    text = "".join([segment.text for segment in segments]).strip()
    print(f"🌍 Detected language '{info.language}' with probability {info.language_probability:.2f}")
    print(f"📝 Transcription: {text}")
    return text

if __name__ == "__main__":
    if not os.path.exists("input.wav"): print("Create 'input.wav' for testing.")
    else: transcription = transcribe_audio()