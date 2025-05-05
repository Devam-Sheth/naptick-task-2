# Naptick AI Challenge - Task 2: Voice-to-Voice Sleep Coaching Agent

## Task Description

The goal was to create a voice-to-voice intelligent agent using a voice-capable model (or simulated system), fine-tuned or adapted for sleep-domain conversations. The agent should understand queries around sleep health and provide improved, relevant responses based on the fine-tuning.

## Features Implemented

* **Voice Input:** Captures user audio via microphone (`audio_input.py`).
* **Transcription:** Converts spoken audio to text using `faster-whisper` (`transcribe.py`).
* **Fine-tuned Language Model:** Fine-tuned EleutherAI/pythia-410m on a sleep-focused conversational dataset (`train_model.py`). The model is hosted on Hugging Face Hub: [devam-sheth-bits/sleep-ai-combined-evolving](https://huggingface.co/devam-sheth-bits/sleep-ai-evolving) 
* **Multi-turn Conversation:** Manages conversation history to provide contextually relevant responses (`main.py`, `inference_sleep_ai.py`).
* **Text-to-Speech Output:** Speaks the assistant's responses using `pyttsx3` (`tts_output.py`).

## Technology Stack

* Python 3.10
* PyTorch
* Hugging Face Libraries: `transformers`, `datasets`, `huggingface_hub`, `accelerate`
* Transcription: `faster-whisper`
* Audio I/O: `sounddevice`, `scipy`
* TTS: `pyttsx3`

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [Link to GitHub Repo]
    cd [repository-folder-name]
    ```
2.  **Create/Activate Environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    # source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    # Ensure pip is up-to-date
    python -m pip install --upgrade pip
    # Install requirements
    pip install -r requirements.txt
    ```
    *(Note: You need to create a `requirements.txt` file listing all dependencies. You can generate one using `pip freeze > requirements.txt` after installing everything in your environment).*
4.  **Hugging Face Login:** Log in to access the fine-tuned model (if private) and potentially push models if you modify the training script.
    ```bash
    huggingface-cli login
    # or C:\path\to\python310\Scripts\huggingface-cli.exe login
    ```
    You'll need a Hugging Face account and a User Access Token with `write` permissions.

## Running the Agent

Execute the main script from the terminal:

```bash
python main.py
# Or specify Python version if needed:
# py -3.10 main.py
