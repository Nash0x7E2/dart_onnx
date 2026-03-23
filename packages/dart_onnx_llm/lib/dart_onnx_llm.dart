/// DartONNX LLM — High-level chat and text generation API for ONNX models.
///
/// Provides a simple, intuitive interface for loading large language models
/// and chatting with them, built on top of [dart_onnx].
///
/// ## Getting Started
///
/// ```dart
/// import 'package:dart_onnx_llm/dart_onnx_llm.dart';
///
/// void main() async {
///   final pipeline = await TextGenerationPipeline.fromDirectory('path/to/model');
///   final session = pipeline.createChatSession(
///     systemPrompt: 'You are a helpful assistant.',
///   );
///
///   final stream = session.sendMessageStream('Hello!');
///   await for (final chunk in stream) {
///     stdout.write(chunk);
///   }
/// }
/// ```
library;

// Config
export 'src/config/model_config.dart';
export 'src/config/tokenizer_config.dart';
export 'src/config/generation_config.dart';

// Tokenizer
export 'src/tokenizer/tokenizer.dart';

// Model
export 'src/model/causal_lm.dart';

// Chat
export 'src/chat/chat_message.dart';
export 'src/chat/chat_session.dart';

// Pipeline
export 'src/pipeline/text_generation_pipeline.dart';

// Sampler
export 'src/sampler/sampler.dart';
