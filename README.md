# TTS LLM Agent

A Python-based conversational AI agent integrating **Text-to-Speech (TTS)**, **Speech-to-Text (STT)**, and a **Large Language Model (LLM)** with smart web search capabilities. The agent, inspired by JARVIS, supports voice and text input, generates spoken responses, and automatically searches the web for queries requiring current information.

## Features
- **Voice Interaction**: Uses STT for voice input and TTS for spoken responses.
- **Smart Web Search**: Automatically performs web searches for queries needing up-to-date information using DuckDuckGo.
- **Interrupt Handling**: Press the `Esc` key to interrupt responses.
- **Streaming Responses**: Real-time text and speech output for a conversational experience.
- **Modular Design**: Separate initialization for TTS, STT, and LLM components for robust error handling.

## Prerequisites
To run this project, ensure you have the following:
- **Python**: Version 3.8 or higher.
- **Conda**: For managing the virtual environment.
- **Hardware**: A microphone for STT and speakers/headphones for TTS.
- **Internet Connection**: Required for web searches and model downloads.

## Setup Instructions

### 1. Clone the Repository
Clone the project repository to your local machine:
```bash
git clone https://github.com/hacktivist211/tts-llm-agent.git
cd tts-llm-agent
```

*Note*: If the repository URL is incorrect, replace `hacktivist211` with the correct GitHub username or repository path.

### 2. Create and Activate a Conda Environment
Create a Conda environment to manage dependencies:
```bash
conda create -n tts-llm-agent python=3.8
conda activate tts-llm-agent
```

### 3. Install Dependencies
Install the required Python packages using `pip`:
```bash
pip install numpy soundfile sounddevice requests beautifulsoup4 tqdm keyboard langchain-ollama
```

#### Additional Dependencies
- **Kokoro TTS**: Install the `kokoro` package for TTS functionality:
  ```bash
  pip install kokoro
  ```
  The `hexgrad/Kokoro-82M` model will be downloaded automatically during initialization.

- **RealtimeSTT**: Install the `RealtimeSTT` package for STT functionality:
  ```bash
  pip install RealtimeSTT
  ```

- **Ollama LLM**: The LLM uses `langchain-ollama` with the `goekdenizguelmez/JOSIEFIED-Qwen3:8b` model. Install Ollama and pull the model:
  1. Install Ollama from [Ollama's official site](https://ollama.ai/).
  2. Pull the model:
     ```bash
     ollama pull goekdenizguelmez/JOSIEFIED-Qwen3:8b
     ```

### 4. Verify Installation
Ensure all dependencies are installed correctly:
```bash
python -c "import numpy, soundfile, sounddevice, requests, bs4, tqdm, keyboard, langchain_ollama, kokoro, RealtimeSTT"
```
If no errors appear, the environment is set up correctly.

## Project Execution

### Running the Agent
1. Activate the Conda environment:
   ```bash
   conda activate tts-llm-agent
   ```

2. Run the main script:
   ```bash
   python tts_llm_agent.py
   ```

3. The agent will initialize the TTS, STT, and LLM modules, displaying a progress bar for TTS loading. Upon completion, it will announce:
   ```
   All systems online! I have smart internet access and will search automatically when your questions need current information. Press esc to interrupt me anytime.
   ```

### Interacting with the Agent
- **Voice Input**: Speak into your microphone. Say "exit" to quit.
- **Text Input**: If STT is unavailable (e.g., no microphone), type commands when prompted.
- **Interrupt Responses**: Press the `Esc` key to stop the agent’s response.
- **Disable Web Search**: Prefix queries with "no search:" to skip web searches (e.g., "no search: tell me a joke").

### Example Commands
- **General Query**: "What's the weather like today?"
  - Triggers a web search and provides spoken/text response.
- **Simple Query**: "Hello"
  - Responds without searching.
- **Math Query**: "What is 5 + 3?"
  - Answers directly.
- **No Search Query**: "no search: What's your name?"
  - Bypasses web search.

## Troubleshooting
- **TTS/STT/LLM Initialization Errors**:
  - Verify all dependencies are installed.
  - Ensure internet connectivity for model downloads.
  - Confirm Ollama is running (`ollama serve`) before starting the script.
- **No Audio Output**:
  - Check that speakers/headphones are connected and not muted.
  - Verify `sounddevice` is configured for your audio device.
- **No Voice Input**:
  - Ensure your microphone is connected and functional.
  - If STT fails, the agent will prompt for text input.
- **Web Search Errors**:
  - Confirm internet connectivity.
  - Ensure DuckDuckGo’s HTML search endpoint is accessible.

## Notes
- The default TTS voice is `bm_lewis`. If unavailable, the script tries alternative voices or a default setting.
- Audio sample rate is 24,000 Hz.
- Threading ensures asynchronous TTS processing for smooth interaction.
- Web searches are triggered for queries needing current information (e.g., news, weather) or complex questions.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


`tts_llm_agent.py` file using:
  ```bash
  git init
  git add tts_llm_agent.py
  git commit -m "Initial commit"
  git remote add origin https://github.com/hacktivist211/tts-llm-agent.git
  git push -u origin main
  ```
