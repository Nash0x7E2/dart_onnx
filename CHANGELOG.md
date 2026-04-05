## 0.1.2

- Added a new high-level Chat API package (`dart_onnx_llm`) with text generation, streaming, and conversation history.
- Implemented `TextGenerationPipeline`, `ChatSession`, and `CausalLM`.
- Added GPT-2 style `ByteLevel` BPE tokenizer with ChatML formatting.
- Added generation samplers (temperature, top-p, top-k, repetition penalty).
- Added `cli_chat.dart` interactive terminal example.
- Enhanced ONNX Runtime library loading with better error hints.

## 0.1.1

- Fixed an issue where model paths were not correctly parsed on Windows.

## 0.1.0

- Initial release.
- Added ONNX Runtime Dart FFI bindings.
- Added `DartONNXSession` and `DartONNXTensor` with optimized memory management.
- Added support for configuring Execution Providers (CoreML, NNAPI, etc.) with automatic fallbacks.
- Added an example demonstrating inference with a Hugging Face ONNX model (SmolLM2-135M).
- Added test suites.
