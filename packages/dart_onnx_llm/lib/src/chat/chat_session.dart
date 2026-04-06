import 'dart:async';

import '../chat/chat_message.dart';
import '../chat/prompt_formatter.dart';
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
  final PromptFormatter _formatter;
  final GenerationConfig _defaultConfig;

  /// Token IDs that signal end-of-turn (e.g. `<|im_end|>`).
  final List<int> _stopTokenIds;

  /// The full conversation history.
  final List<ChatMessage> _messages = [];

  ChatSession({
    required CausalLM model,
    required Tokenizer tokenizer,
    required PromptFormatter formatter,
    String? systemPrompt,
    GenerationConfig config = const GenerationConfig(),
    List<int> stopTokenIds = const [],
  }) : _model = model,
       _tokenizer = tokenizer,
       _formatter = formatter,
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
    final prompt = _formatter.format(_messages);

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
}
