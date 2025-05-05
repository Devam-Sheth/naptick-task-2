# voice_agent/tts_output.py
import pyttsx3
import traceback

engine = pyttsx3.init()
engine.setProperty('rate', 165)
print("TTS Engine Initialized.")

def speak_text(text):
    if engine is None: print("ERROR: TTS engine not initialized."); return
    if not text or not isinstance(text, str): print("Warning: Invalid text for TTS."); return

    print("🗣️ Speaking...")
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    test_text = "This is a test of the text to speech system."; print(f"Testing TTS: '{test_text}'")
    speak_text(test_text); print("TTS test complete.")