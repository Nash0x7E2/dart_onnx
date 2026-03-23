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

  /// The full conversation history.
  final List<ChatMessage> _messages = [];

  ChatSession({
    required CausalLM model,
    required Tokenizer tokenizer,
    String? systemPrompt,
    GenerationConfig config = const GenerationConfig(),
  }) : _model = model,
       _tokenizer = tokenizer,
       _defaultConfig = config {
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

    // Append the user message to history.
    _messages.add(ChatMessage.user(message));

    // Format the full conversation into a prompt string.
    final prompt = _formatPrompt(_messages);

    // Encode.
    final inputIds = _tokenizer.encode(prompt);

    // Generate tokens.
    final sampler = Sampler(config);
    final responseBuffer = StringBuffer();

    await for (final tokenId in _model.generate(
      inputIds,
      config: config,
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
    if (keepSystemPrompt && _messages.isNotEmpty && _messages.first.role == ChatRole.system) {
      final system = _messages.first;
      _messages.clear();
      _messages.add(system);
    } else {
      _messages.clear();
    }
  }

  /// Formats a list of messages into a prompt string.
  ///
  /// Uses the chat template from [TokenizerConfig] if available, otherwise
  /// falls back to a simple ChatML format.
  String _formatPrompt(List<ChatMessage> messages) {
    // TODO(nash): Use Jinja-style chat template from tokenizer_config.json
    // when available. For now, fall back to ChatML.
    final buffer = StringBuffer();
    for (final msg in messages) {
      buffer.writeln('<|im_start|>${msg.role.name}');
      buffer.writeln(msg.content);
      buffer.writeln('<|im_end|>');
    }
    buffer.writeln('<|im_start|>assistant');
    return buffer.toString();
  }
}
