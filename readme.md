# TeeleAgentHF

TeeleAgentHF is an AI-powered telecalling agent built for a Hugging Face competition. It captures live audio, transcribes speech, extracts scheduling intent, evaluates feasibility, and confirms bookings. Designed for low-VRAM deployment (4GB budget) and Hugging Face Spaces.

## Key Features

- Real-time microphone capture with Gradio UI
- ASR: Cohere Transcribe (streaming)
- Intent parsing: Qwen2.5-7B-Instruct (GGUF via llama-cpp-python)
- Evaluation: MiniCPM3-4B (int4 quantized evaluator)
- VAD: Silero VAD (ONNX)
- Persistent bookings in SQLite (`data/calls.db`)
- Scheduling rules and slot-checking logic

## Architecture

- app.py: Gradio front-end and session controls
- pipeline/: transcriber, intent parser, evaluator, orchestrator, VAD listener
- config.py & hf_config.json: model and inference configuration
- data/calls.db and db.py: call logging and booking persistence

## Requirements

- Python 3.10+ (3.11 recommended)
- CUDA-capable GPU for llama-cpp-python Qwen inference (recommended)
- Install dependencies: `pip install -r requirements.txt`
- Note: `llama-cpp-python` may require a CUDA-enabled build. Example:
  ```bash
  CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUBLAS=on" pip install -U "llama-cpp-python"
  ```

## Running Locally

1. Create and activate a virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure models referenced in `hf_config.json` are available or accessible via Hugging Face
4. Start the app:
   ```bash
   python app.py
   ```
5. Open http://127.0.0.1:7860 in a browser

## Deployment (Hugging Face Spaces)

- Ensure `app.py` listens on 0.0.0.0:7860 (config.py already uses these defaults)
- Provide model files or configure download/autoload in `hf_config.json`
- Verify VRAM budget and use quantized GGUF models to fit resource limits

## Configuration

- Edit `config.py` and `hf_config.json` to tune models, quantization, batch sizes, and scheduling rules (working hours, slot lengths, etc.)

## Collaborators

- Saurav Kumar Yadav <sauravkumaryadav100@gmail.com>

## Contributing

- Open issues or PRs. For large model changes, include resource and runtime notes.

## License

See LICENSE in the repository root.

## Contact

For questions about this project, contact the repository owner or listed collaborators.
