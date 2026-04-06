import 'dart:io';

import 'package:dart_onnx/dart_onnx.dart';

import '../chat/chat_session.dart';
import '../chat/prompt_formatter.dart';
import '../config/generation_config.dart';
import '../config/model_config.dart';
import '../config/tokenizer_config.dart';
import '../model/causal_lm.dart';
import '../tokenizer/tokenizer.dart';

/// The high-level entry point for text generation with ONNX-based LLMs.
///
/// Loads the model, tokenizer, and configuration from a model directory
/// and provides simple APIs for chat sessions and text generation.
///
/// ## Basic Usage
///
/// ```dart
/// final pipeline = await TextGenerationPipeline.fromDirectory('path/to/model');
/// final session = pipeline.createChatSession(
///   systemPrompt: 'You are a helpful assistant.',
/// );
///
/// final stream = session.sendMessageStream('Hello!');
/// await for (final chunk in stream) {
///   stdout.write(chunk);
/// }
/// ```
class TextGenerationPipeline {
  /// The underlying causal language model.
  final CausalLM model;

  /// The tokenizer for encoding/decoding text.
  final Tokenizer tokenizer;

  /// The model configuration.
  final ModelConfig modelConfig;

  /// The tokenizer configuration (chat templates, special tokens).
  final TokenizerConfig? tokenizerConfig;

  TextGenerationPipeline({
    required this.model,
    required this.tokenizer,
    required this.modelConfig,
    this.tokenizerConfig,
  });

  /// Loads a [TextGenerationPipeline] from a directory containing
  /// model files.
  ///
  /// Expected directory structure for monolithic models (e.g. SmolLM2):
  /// ```
  /// model_dir/
  /// ├── config.json
  /// ├── tokenizer.json
  /// └── model.onnx
  /// ```
  ///
  /// Expected directory structure for split-graph models (e.g. Gemma 4):
  /// ```
  /// model_dir/
  /// ├── config.json
  /// ├── tokenizer.json
  /// ├── embed_tokens.onnx
  /// └── decoder_model_merged.onnx
  /// ```
  static Future<TextGenerationPipeline> fromDirectory(
    String directoryPath, {
    String? modelFileName,
    List<DartONNXExecutionProvider> executionProviders = const [
      DartONNXExecutionProvider.cpu,
    ],
    DartONNXLoggingLevel loggingLevel = DartONNXLoggingLevel.warning,
  }) async {
    final dir = Directory(directoryPath);
    if (!dir.existsSync()) {
      throw ArgumentError('Model directory not found: $directoryPath');
    }

    // Load config.json
    final configPath = '${dir.path}/config.json';
    if (!File(configPath).existsSync()) {
      throw ArgumentError('config.json not found in: $directoryPath');
    }
    final modelConfig = await ModelConfig.fromFile(configPath);

    // Load tokenizer_config.json (optional)
    TokenizerConfig? tokenizerConfig;
    final tokenizerConfigPath = '${dir.path}/tokenizer_config.json';
    if (File(tokenizerConfigPath).existsSync()) {
      tokenizerConfig = await TokenizerConfig.fromFile(tokenizerConfigPath);
    }

    // Load tokenizer.json
    final tokenizerPath = '${dir.path}/tokenizer.json';
    if (!File(tokenizerPath).existsSync()) {
      throw ArgumentError('tokenizer.json not found in: $directoryPath');
    }
    final tokenizer = await Tokenizer.fromFile(
      tokenizerPath,
      config: tokenizerConfig,
    );

    // Detect architecture: Split-Graph vs Monolithic
    final decoderMergedPath = '${dir.path}/decoder_model_merged.onnx';
    final embedTokensPath = '${dir.path}/embed_tokens.onnx';
    
    CausalLM model;

    if (File(decoderMergedPath).existsSync() && File(embedTokensPath).existsSync()) {
      // Split-graph architecture (Gemma 4)
      model = CausalLM.fromPipeline(
        decoderMergedPath,
        embedPath: embedTokensPath,
        config: modelConfig,
        executionProviders: executionProviders,
        loggingLevel: loggingLevel,
      );
    } else {
      // Monolithic architecture (SmolLM2, Llama)
      final modelPath = _resolveModelPath(dir, modelFileName);
      model = CausalLM.fromFile(
        modelPath,
        config: modelConfig,
        executionProviders: executionProviders,
        loggingLevel: loggingLevel,
      );
    }

    return TextGenerationPipeline(
      model: model,
      tokenizer: tokenizer,
      modelConfig: modelConfig,
      tokenizerConfig: tokenizerConfig,
    );
  }

  /// Creates a stateful [ChatSession] for multi-turn conversations.
  ///
  /// Automatically configures stop tokens from the tokenizer config so
  /// generation halts at the end-of-turn marker (e.g. `<|im_end|>`).
  ChatSession createChatSession({
    String? systemPrompt,
    GenerationConfig config = const GenerationConfig(),
  }) {
    // Resolve stop token IDs from the tokenizer config.
    final stopTokenIds = <int>[];
    final eosToken = tokenizerConfig?.eosToken;
    if (eosToken != null) {
      final eosId = tokenizer.tokenToId(eosToken);
      if (eosId != null) {
        stopTokenIds.add(eosId);
      }
    }

    final formatter = PromptFormatter.fromTemplate(tokenizerConfig?.chatTemplate);

    return ChatSession(
      model: model,
      tokenizer: tokenizer,
      formatter: formatter,
      systemPrompt: systemPrompt,
      config: config,
      stopTokenIds: stopTokenIds,
    );
  }

  /// Generates text from a raw prompt string, streaming the response.
  Stream<String> generateStream(
    String prompt, {
    GenerationConfig config = const GenerationConfig(),
  }) async* {
    final inputIds = tokenizer.encode(prompt);

    await for (final tokenId in model.generate(inputIds, config: config)) {
      yield tokenizer.decode([tokenId]);
    }
  }

  /// Generates text from a raw prompt string, returning the full result.
  Future<String> generate(
    String prompt, {
    GenerationConfig config = const GenerationConfig(),
  }) async {
    final buffer = StringBuffer();
    await for (final chunk in generateStream(prompt, config: config)) {
      buffer.write(chunk);
    }
    return buffer.toString();
  }

  /// Disposes the underlying model resources.
  void dispose() {
    model.dispose();
  }

  /// Resolves the path to the ONNX model file.
  static String _resolveModelPath(Directory dir, String? modelFileName) {
    if (modelFileName != null) {
      final path = '${dir.path}/$modelFileName';
      if (!File(path).existsSync()) {
        throw ArgumentError('Model file not found: $path');
      }
      return path;
    }

    // Auto-detect: prefer quantized, then any .onnx file.
    final quantizedPath = '${dir.path}/model_quantized.onnx';
    if (File(quantizedPath).existsSync()) return quantizedPath;

    final defaultPath = '${dir.path}/model.onnx';
    if (File(defaultPath).existsSync()) return defaultPath;

    // Search for any .onnx file.
    final onnxFiles = dir
        .listSync()
        .whereType<File>()
        .where((f) => f.path.endsWith('.onnx'))
        .toList();

    if (onnxFiles.isEmpty) {
      throw ArgumentError('No .onnx model file found in: ${dir.path}');
    }

    return onnxFiles.first.path;
  }
}
