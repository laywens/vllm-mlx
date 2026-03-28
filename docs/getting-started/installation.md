# Installation

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.10+

## Install with uv (Recommended)

```bash
git clone https://github.com/waybarrios/vllm-mlx.git
cd vllm-mlx

uv pip install -e .
```

For the reviewed Apple Silicon verification environment used by CI
(`python 3.11` on `arm64`) from a trusted checkout:

```bash
uv pip install --require-hashes --only-binary=:all: -r requirements/locks/local-verify-vision.txt
uv pip install --no-deps --no-build-isolation -e .
```

## Install with pip

```bash
git clone https://github.com/waybarrios/vllm-mlx.git
cd vllm-mlx

pip install -e .
```

For the tracked dependency policy used in CI and reviewed contributor environments, see [Dependency Security](../development/dependency-security.md).

### Optional: Vision Support

For video processing with transformers:

```bash
pip install -e ".[vision]"
```

### Optional: Audio Support (STT/TTS)

```bash
pip install mlx-audio
```

### Optional: Embeddings

```bash
pip install mlx-embeddings
```

## What Gets Installed

- `mlx`, `mlx-lm`, `mlx-vlm` - MLX framework and model libraries
- `transformers`, `tokenizers` - HuggingFace libraries
- `opencv-python` - Video processing
- `gradio` - Chat UI
- `psutil` - Resource monitoring
- `mlx-audio` (optional) - Speech-to-Text and Text-to-Speech
- `mlx-embeddings` (optional) - Text embeddings

## Verify Installation

```bash
# Check CLI commands
vllm-mlx --help
vllm-mlx-bench --help
vllm-mlx-chat --help

# Test with a small model
vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit --prompts 1
```

## Troubleshooting

### MLX not found

Ensure you're on Apple Silicon:
```bash
uname -m  # Should output "arm64"
```

### Model download fails

Check your internet connection and HuggingFace access. Some models require authentication:
```bash
huggingface-cli login
```

### Out of memory

Use a smaller quantized model:
```bash
vllm-mlx serve mlx-community/Llama-3.2-1B-Instruct-4bit
```
