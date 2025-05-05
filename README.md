# Naptick AI Challenge - Task 2: Voice-to-Voice Sleep Coaching Agent

## Task Description

The goal was to create a voice-to-voice intelligent agent using a simulated voice system and a fine-tuned language model. The agent is adapted for sleep-domain conversations, understands related queries, and provides relevant multi-turn responses.

## Features Implemented

* **Voice Input:** Captures user audio via microphone (`audio_input.py`).
* **Transcription:** Converts spoken audio to text using `faster-whisper` (`transcribe.py`).
* **Fine-tuned Language Model:** Fine-tuned `EleutherAI/pythia-410m` on combined sleep conversation datasets.
* **Multi-turn Conversation:** Manages conversation history for contextual responses (`main.py`, `inference_sleep_ai.py`).
* **Text-to-Speech Output:** Speaks the assistant's responses using `pyttsx3` (`tts_output.py`).
* **Hugging Face Hub Integration:** The fine-tuned model is loaded directly from Hugging Face Hub.

## Fine-tuned Model on Hub

The core fine-tuned language model used by this agent is hosted on Hugging Face Hub and **will be downloaded automatically** when the application runs for the first time.

* **Model ID:** [`devam-sheth-bits/sleep-ai-evolving`](https://huggingface.co/devam-sheth-bits/sleep-ai-evolving)

## Technology Stack

* Python 3.10
* PyTorch
* Hugging Face Libraries: `transformers`, `datasets`, `huggingface_hub`, `accelerate`
* Transcription: `faster-whisper`
* Audio I/O: `sounddevice`, `scipy`
* TTS: `pyttsx3`

## Setup and Installation Instructions (For Running the Agent)

Follow these steps carefully to set up and run the voice agent on another device:

1.  **Clone the Repository:**
    ```bash
    git clone [Link to Your GitHub Repository]
    cd [repository-folder-name]
    ```

2.  **Create and Activate Virtual Environment (Highly Recommended):**
    ```bash
    # Create environment (e.g., named .venv)
    python -m venv .venv

    # Activate environment
    # Windows (PowerShell/CMD):
    .\.venv\Scripts\activate
    # macOS/Linux (Bash/Zsh):
    # source .venv/bin/activate
    ```
    *(Note: The `.venv` directory should NOT be committed to Git).*

3.  **Install Python Dependencies:**
    Install all required Python libraries using the provided `requirements.txt` file:
    ```bash
    # Ensure pip is up-to-date within the venv
    python -m pip install --upgrade pip
    # Install requirements
    pip install -r requirements.txt
    ```
    *(Note: You need to create the `requirements.txt` file in your repository using `pip freeze > requirements.txt` from your activated environment after installing everything).*

4.  **Install System Dependencies (If Necessary):**
    * **`sounddevice`:** May require system libraries like `portaudio`.
        * On Debian/Ubuntu Linux: `sudo apt-get update && sudo apt-get install libportaudio2`
        * On macOS (using Homebrew): `brew install portaudio`
        * On Windows: Usually works out-of-the-box or with specific audio driver installations.
    * **`pyttsx3`:** May require OS-level TTS engines.
        * On Windows: Uses SAPI5 (usually built-in).
        * On macOS: Uses NSSpeechSynthesizer (built-in).
        * On Linux: May require `espeak` (`sudo apt-get update && sudo apt-get install espeak`).
    * *Consult the documentation for `sounddevice` and `pyttsx3` if audio input or output fails.*

5.  **Hugging Face Login (Required):**
    You need to log in to your Hugging Face account via the terminal. This allows the application to download the fine-tuned model and associated tokenizer from the Hub.
    ```bash
    huggingface-cli login
    ```
    Enter your Hugging Face User Access Token when prompted (you need one with at least `read` permissions).

## Running the Agent

Once the setup is complete:

1.  **Ensure your microphone and speakers are configured correctly.**
2.  **Run the main script** from your terminal (make sure your virtual environment is activated):
    ```bash
    python main.py
    ```
    *(Or use `py -3.10 main.py` if you need to specify the interpreter)*.
3.  **First Run Note:** The first time you run it, the script will need to download:
    * The fine-tuned LLM (`devam-sheth-bits/sleep-ai-evolving` ~900MB+).
    * The `faster-whisper` model (`base` model, ~140MB).
    * The embedding model used by `transcribe.py` if applicable (though `faster-whisper` handles this internally).
    This might take several minutes depending on your internet speed. Subsequent runs will use the cached models.
4.  **Interact:** The assistant will greet you. Speak clearly when prompted (`🎙️ Recording... Speak now.`). Use exit phrases like "goodbye" or "exit" to end the conversation.

## File Structure (Included in Repo)

* `main.py`: Main application loop.
* `inference_sleep_ai.py`: Loads the fine-tuned model *from Hub* and generates responses.
* `audio_input.py`: Handles microphone recording.
* `transcribe.py`: Transcribes audio using Faster Whisper.
* `tts_output.py`: Handles text-to-speech.
* `train_model.py`: Script used for fine-tuning (provided for reference).
* `multichat_dataset.py`: Script used for generating initial data (provided for reference).
* `*.jsonl`, `*.csv`: Data files used for training (included for reproducibility).
* `requirements.txt`: Python dependencies.
* `README.md`: This file.
* `sample_conversations/`: Folder containing audio samples.

## Excluded Files (Not in Repo)

* `.venv/`: Python virtual environment files.
* `training_results*/`: Checkpoints saved locally during training.
* `*_final/`: Final model saved locally after training.
* `__pycache__/`: Python bytecode cache directories.
* `*.wav`: Temporary audio files generated during runtime (like `temp_user_input.wav`).
    *(It's recommended to use a `.gitignore` file to automatically exclude these).*

## Limitations & Future Work

* The model's ability to provide truly "evolving" advice is dependent on the quality and quantity of the `evolving_advice_200.jsonl` data used. More sophisticated data generation and potentially larger models would improve this.
* Relies on external library quality for Transcription/TTS.
* CPU inference can be slow for response generation. GPU would improve speed.
* Error handling can be further enhanced.
