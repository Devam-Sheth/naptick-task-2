# voice_agent/main.py
# Multi-turn loop - NO ERROR HANDLING. Compatible with inference_sleep_ai.py

import time
import os
import traceback
from audio_input import record_audio
from transcribe import transcribe_audio
from inference_sleep_ai import query_sleep_ai_multi_turn
from tts_output import speak_text

# --- Configuration ---
RECORDING_DURATION = 7
AUDIO_FILENAME = "temp_user_input.wav"
MAX_HISTORY_TURNS = 10
EXIT_KEYWORDS = ["exit", "quit", "stop", "goodbye", "bye bye"]

def run_conversation():
    conversation_history = []
    welcome_message = "Hello! I am your Sleep AI assistant. How can I help you today? Say 'exit' or 'goodbye' to end our chat."
    print(f"🤖 Assistant: {welcome_message}")
    speak_text(welcome_message)

    while True:
        print("-" * 20)
        record_audio(filename=AUDIO_FILENAME, duration=RECORDING_DURATION)
        user_text = transcribe_audio(filepath=AUDIO_FILENAME)
        if not user_text or user_text.strip() == "" or user_text.lower().strip().startswith("error"):
            print(f"Transcription issue or silence: {user_text}. Please try again.")
            continue
        print(f"👤 You (transcribed): {user_text}")

        cleaned = ''.join(c for c in user_text.strip().lower() if c.isalnum() or c.isspace()).strip()
        should_exit = any(k == cleaned or f" {k} " in f" {cleaned} " or cleaned.startswith(k) or cleaned.endswith(k) for k in EXIT_KEYWORDS)
        if should_exit:
            farewell = "Okay, ending now. Sleep well!"; print(f"🤖: {farewell}")
            speak_text(farewell); break

        conversation_history.append({"role": "user", "content": user_text})
        if MAX_HISTORY_TURNS > 0 and len(conversation_history) > MAX_HISTORY_TURNS * 2:
            print("(Trimming history...)"); conversation_history = conversation_history[-(MAX_HISTORY_TURNS * 2):]

        assistant_reply = query_sleep_ai_multi_turn(conversation_history)
        if not assistant_reply:
            print("Warn: AI no response."); conversation_history.pop(); continue

        conversation_history.append({"role": "assistant", "content": assistant_reply})
        print(f"🤖 Assistant: {assistant_reply}")
        speak_text(assistant_reply)

    print("-" * 20); print("Conversation ended.")
    if os.path.exists(AUDIO_FILENAME):
        os.remove(AUDIO_FILENAME); print(f"Removed temp file: {AUDIO_FILENAME}")

if __name__ == "__main__":
    run_conversation()