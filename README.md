# Naptick AI Challenge - Task 2: Voice-to-Voice Sleep Coaching Agent

**GitHub Repository:** [https://github.com/Devam-Sheth/naptick-task-2](https://github.com/Devam-Sheth/naptick-task-2)

## Task Description

The objective of Task 2 was to create a voice-to-voice intelligent agent, fine-tuned or adapted for sleep-domain conversations using relevant datasets. The agent is expected to understand user queries around sleep health, routines, improvements, and advice, providing improved and specialized responses as a result of the fine-tuning. This project implements such an agent through a modular system.

## Features Implemented

* **Voice Input:** Captures user audio via microphone (`audio_input.py`).
* **Transcription:** Converts spoken audio to text using the `faster-whisper` library (`transcribe.py`).
* **Fine-tuned Language Model:** An `EleutherAI/pythia-410m` model was fine-tuned on a custom, comprehensive dataset focused on sleep coaching conversations (`train_model.py`).
* **Multi-turn Conversation:** The agent manages conversation history to provide contextually relevant responses over multiple turns (`main.py`, `inference_sleep_ai.py`).
* **Text-to-Speech Output:** The assistant's textual responses are converted to audible speech using `pyttsx3` (`tts_output.py`).
* **Hugging Face Hub Integration:** The fine-tuned model is hosted on and loaded from Hugging Face Hub.

## Fine-tuned Model on Hugging Face Hub

The core of this agent is a language model fine-tuned specifically for sleep coaching dialogues.
* **Model ID:** `devam-sheth-bits/enhanced-sleep-ai`
* **Model Link:** [https://huggingface.co/devam-sheth-bits/enhanced-sleep-ai](https://huggingface.co/devam-sheth-bits/enhanced-sleep-ai)

This model will be downloaded automatically when the application is run for the first time (if not already cached).

## Technology Stack

* **Programming Language:** Python 3.10
* **Core ML/NLP Libraries:** PyTorch, Hugging Face (`transformers`, `datasets`, `huggingface_hub`, `accelerate`)
* **Transcription:** `faster-whisper` (utilizing the `base` model)
* **Audio Handling:** `sounddevice` (for microphone input), `scipy` (for WAV file operations)
* **Text-to-Speech (TTS):** `pyttsx3`

## Setup and Installation Instructions

To set up and run this project locally, please follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Devam-Sheth/naptick-task-2.git
    cd naptick-task-2
    ```

2.  **Create and Activate a Python Virtual Environment (Highly Recommended):**
    ```bash
    python -m venv .venv
    # On Windows (PowerShell/CMD):
    .\.venv\Scripts\activate
    # On macOS/Linux (Bash/Zsh):
    # source .venv/bin/activate
    ```

3.  **Install Python Dependencies:**
    Ensure `pip` is up-to-date and then install the required packages from `requirements.txt`.
    ```bash
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

4.  **Install System Dependencies (If Necessary):**
    * **`sounddevice`:** This library interfaces with your system's audio capabilities and might require `portaudio`.
        * On Debian/Ubuntu Linux: `sudo apt-get update && sudo apt-get install libportaudio2`
        * On macOS (using Homebrew): `brew install portaudio`
        * On Windows: Usually works directly, but ensure your microphone drivers are up-to-date.
    * **`pyttsx3`:** This library relies on OS-level TTS engines.
        * On Windows: Uses SAPI5 (typically built-in).
        * On macOS: Uses NSSpeechSynthesizer (built-in).
        * On Linux: May require `espeak` (`sudo apt-get update && sudo apt-get install espeak`).
    * If you encounter issues with audio input or output, please consult the documentation for these libraries.

5.  **Hugging Face Login (Required):**
    You must log in to your Hugging Face account via the terminal. This allows the application to download the fine-tuned model (`devam-sheth-bits/enhanced-sleep-ai`) and its tokenizer from the Hub.
    ```bash
    huggingface-cli login
    ```
    You will be prompted to enter a Hugging Face User Access Token. Ensure the token has at least `read` permissions.

## Running the Agent

Once the setup is complete:

1.  Ensure your microphone is connected and selected as the default input device, and your speakers/headphones are working.
2.  Navigate to the root of the cloned repository in your terminal (where `main.py` is located), with your virtual environment activated.
3.  Execute the main script:
    ```bash
    python main.py
    ```
    *(Or `py -3.10 main.py` if you need to specify your Python 3.10 interpreter explicitly).*

4.  **First Run Considerations:**
    * The first time you run `main.py`, the system will download the fine-tuned language model (`devam-sheth-bits/enhanced-sleep-ai`, approx. 900MB+) and the `faster-whisper` base model (approx. 140MB) if they are not already cached locally. This initial download process may take several minutes depending on your internet connection speed.
    * Subsequent runs will be faster as these models will be loaded from the local cache.

5.  **Interacting with the Agent:**
    * The assistant will provide an initial greeting.
    * When you see `🎙️ Recording... Speak now.` in the terminal, speak your query clearly into the microphone.
    * The agent will transcribe your speech, process it, generate a response, and speak the response back to you.
    * To end the conversation, use phrases like "goodbye," "exit," or "quit."

## Fine-tuning Details

* **Base Model:** `EleutherAI/pythia-410m`.
* **Training Dataset:** The fine-tuning process utilized a custom dataset of approximately **520 conversational examples specifically focused on sleep coaching scenarios**. This dataset was primarily generated using **Google AI Studio** to create diverse and relevant dialogues for adapting the model to the sleep domain. *(If you also combined this with `dataset1.jsonl` or `dataset2.jsonl`, briefly mention that too, e.g., "This was further combined with programmatically generated conversational flows to ensure coverage of basic interactions.")*
* **Data Formatting:** Conversations were structured with clear roles (`user:`, `assistant:`) and end-of-sequence tokens (`<eos>`) to teach the model turn-taking and appropriate response generation.
* **Training Process:** The fine-tuning was conducted using the script `train_model.py`, leveraging the Hugging Face `Trainer` API, with standard hyperparameters (e.g., AdamW optimizer, learning rate of 2e-5, 1.5 epochs).

## File Structure (Key Files in Repository)

* `main.py`: Orchestrates the voice agent's operation (audio input, transcription, inference, TTS output, conversation loop).
* `train_model.py`: Script used for fine-tuning the language model on the custom sleep coaching dataset.
* `inference_sleep_ai.py`: Contains the logic to load the fine-tuned model from Hugging Face Hub and generate responses based on conversation history.
* `audio_input.py`: Manages recording audio from the microphone.
* `transcribe.py`: Handles speech-to-text using `faster-whisper`.
* `tts_output.py`: Provides text-to-speech functionality using `pyttsx3`.
* `[your_dataset_name.jsonl]`: The primary JSONL file containing the ~520 conversational examples used for fine-tuning. *(Make sure to name this file correctly if you upload it).*
* `requirements.txt`: Lists all Python dependencies required for the project.
* `README.md`: This file.
* `sampleaudio/`: This folder contain the 5 audio/transcript samples demonstrating the agent's capabilities (as per challenge requirements).

## Excluded Files (Not in GitHub Repository)

The following are standard exclusions and are not version-controlled:

* `.venv/`: Python virtual environment directory.
* `training_results*/` (or similar): Local checkpoints saved during model training.
* `*_final/` (or similar): Final model saved locally after training.
* `__pycache__/`: Python bytecode cache directories.
* `*.wav` (e.g., `temp_user_input.wav`): Temporary audio files created during runtime.

*(It is recommended to use a `.gitignore` file to ensure these are not accidentally tracked).*

## Limitations and Potential Future Work

* **Depth of Advice:** While the model is fine-tuned for sleep coaching, the depth and breadth of its advice are limited by the scope of the ~520 training examples.
* **Evolving Advice:** The current implementation focuses on coherent multi-turn dialogue. Truly "evolving" advice that adapts dynamically over very long conversations based on subtle cues would require more sophisticated training data and potentially a larger model.
* **Robustness:** Further work could enhance error handling, particularly for audio I/O and transcription variations.
* **Performance:** Inference on CPU can be slow. GPU acceleration would significantly improve response times.
* **Alternative TTS/STT:** Exploring cloud-based TTS/STT services could improve the naturalness and accuracy of the voice interaction.
