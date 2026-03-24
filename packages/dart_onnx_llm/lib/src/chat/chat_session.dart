import 'dart:async';

import '../chat/chat_message.dart';
import '../config/generation_config.dart';
import '../model/causal_lm.dart';
import '../sampler/sampler.dart';
import '../tokenizer/tokenizer.dart';

/// A stateful chat session that maintains conversation history.
///
/// Created via [TextGenerationPipeline.createChatSession].
///
/// ```dart
/// final session = pipeline.createChatSession(
///   systemPrompt: 'You are a helpful assistant.',
/// );
///
/// final stream = session.sendMessageStream('Hello!');
/// await for (final chunk in stream) {
///   stdout.write(chunk);
/// }
/// ```
class ChatSession {
  final CausalLM _model;
  final Tokenizer _tokenizer;
  final GenerationConfig _defaultConfig;

  /// Token IDs that signal end-of-turn (e.g. `<|im_end|>`).
  final List<int> _stopTokenIds;

  /// The full conversation history.
  final List<ChatMessage> _messages = [];

  ChatSession({
    required CausalLM model,
    required Tokenizer tokenizer,
    String? systemPrompt,
    GenerationConfig config = const GenerationConfig(),
    List<int> stopTokenIds = const [],
  }) : _model = model,
       _tokenizer = tokenizer,
       _defaultConfig = config,
       _stopTokenIds = stopTokenIds {
    if (systemPrompt != null) {
      _messages.add(ChatMessage.system(systemPrompt));
    }
  }

  /// The current conversation history (unmodifiable view).
  List<ChatMessage> get messages => List.unmodifiable(_messages);

  /// Sends a user [message] and streams back the assistant's response
  /// token-by-token as decoded text chunks.
  ///
  /// The user message and the final assistant response are automatically
  /// appended to the conversation [messages] history.
  Stream<String> sendMessageStream(
    String message, {
    GenerationConfig? config,
  }) async* {
    config ??= _defaultConfig;

    // Merge session-level stop tokens with any config-level ones.
    final effectiveConfig = _stopTokenIds.isNotEmpty
        ? GenerationConfig(
            maxTokens: config.maxTokens,
            temperature: config.temperature,
            topP: config.topP,
            topK: config.topK,
            repetitionPenalty: config.repetitionPenalty,
            stopTokenIds: [...?config.stopTokenIds, ..._stopTokenIds],
          )
        : config;

    // Append the user message to history.
    _messages.add(ChatMessage.user(message));

    // Format the full conversation into a prompt string.
    final prompt = _formatPrompt(_messages);

    // Encode.
    final inputIds = _tokenizer.encode(prompt);

    // Generate tokens.
    final sampler = Sampler(effectiveConfig);
    final responseBuffer = StringBuffer();

    await for (final tokenId in _model.generate(
      inputIds,
      config: effectiveConfig,
      sampler: sampler,
    )) {
      final decoded = _tokenizer.decode([tokenId]);
      responseBuffer.write(decoded);
      yield decoded;
    }

    // Append the full assistant response to history.
    _messages.add(ChatMessage.assistant(responseBuffer.toString()));
  }

  /// Sends a user [message] and returns the full assistant response.
  Future<String> sendMessage(String message, {GenerationConfig? config}) async {
    final buffer = StringBuffer();
    await for (final chunk in sendMessageStream(message, config: config)) {
      buffer.write(chunk);
    }
    return buffer.toString();
  }

  /// Clears the conversation history, optionally keeping the system prompt.
  void clearHistory({bool keepSystemPrompt = true}) {
    if (keepSystemPrompt &&
        _messages.isNotEmpty &&
        _messages.first.role == ChatRole.system) {
      final system = _messages.first;
      _messages.clear();
      _messages.add(system);
    } else {
      _messages.clear();
    }
  }

  /// Formats a list of messages into a prompt string using the ChatML format.
  ///
  /// Produces the exact format expected by SmolLM2-Instruct (and other
  /// ChatML-based models):
  ///
  /// ```
  /// <|im_start|>system
  /// You are a helpful assistant.<|im_end|>
  /// <|im_start|>user
  /// Hello!<|im_end|>
  /// <|im_start|>assistant
  /// ```
  String _formatPrompt(List<ChatMessage> messages) {
    // TODO(nash): Use Jinja-style chat template from tokenizer_config.json
    // when available. For now, use the ChatML format directly.
    final buffer = StringBuffer();
    for (final msg in messages) {
      buffer.write('<|im_start|>${msg.role.name}\n');
      buffer.write('${msg.content}<|im_end|>\n');
    }
    buffer.write('<|im_start|>assistant\n');
    return buffer.toString();
  }
}
