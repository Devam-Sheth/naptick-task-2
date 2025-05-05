# voice_agent/audio_input.py
import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="input.wav", duration=5, fs=16000):
    print("🎙️ Recording... Speak now.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, fs, audio)
    print(f"✅ Recording saved to {filename}")

if __name__ == "__main__":
    record_audio()