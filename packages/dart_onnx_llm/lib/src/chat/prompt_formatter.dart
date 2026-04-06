import '../chat/chat_message.dart';

/// Formats a list of [ChatMessage]s into the model-specific prompt string.
///
/// Different model families expect different prompt templates:
///   - **ChatML** (SmolLM2, Llama-Instruct): `<|im_start|>role\ncontent<|im_end|>`
///   - **Gemma** (Gemma 3/4): `<start_of_turn>role\ncontent<end_of_turn>`
///
/// Use [PromptFormatter.fromTemplate] to auto-detect the correct formatter
/// from a `chat_template` string.
abstract class PromptFormatter {
  const PromptFormatter();

  /// Auto-detect the correct formatter from a `chat_template` string.
  ///
  /// Looks for model-specific markers in the template:
  /// - `<start_of_turn>` → Gemma format
  /// - `<|im_start|>` → ChatML format
  ///
  /// Falls back to [ChatMLFormatter] if no markers are found.
  factory PromptFormatter.fromTemplate(String? chatTemplate) {
    if (chatTemplate == null) return const ChatMLFormatter();

    if (chatTemplate.contains('<start_of_turn>')) {
      return const GemmaFormatter();
    }
    return const ChatMLFormatter();
  }

  /// Formats a conversation into the model's expected prompt string.
  ///
  /// The returned string should end with the assistant's turn marker so
  /// the model can begin generating.
  String format(List<ChatMessage> messages);
}

/// Formats messages using the ChatML template.
///
/// Used by SmolLM2-Instruct, Llama-3-Instruct, and other ChatML models.
///
/// ```
/// <|im_start|>system
/// You are a helpful assistant.<|im_end|>
/// <|im_start|>user
/// Hello!<|im_end|>
/// <|im_start|>assistant
/// ```
class ChatMLFormatter extends PromptFormatter {
  const ChatMLFormatter();

  @override
  String format(List<ChatMessage> messages) {
    final buffer = StringBuffer();
    for (final msg in messages) {
      buffer.write('<|im_start|>${msg.role.name}\n');
      buffer.write('${msg.content}<|im_end|>\n');
    }
    buffer.write('<|im_start|>assistant\n');
    return buffer.toString();
  }
}

/// Formats messages using the Gemma chat template.
///
/// Used by Gemma 3 and Gemma 4 models.
///
/// ```
/// <start_of_turn>user
/// Hello!<end_of_turn>
/// <start_of_turn>model
/// ```
///
/// Note: Gemma uses `model` instead of `assistant` for the AI turn.
class GemmaFormatter extends PromptFormatter {
  const GemmaFormatter();

  @override
  String format(List<ChatMessage> messages) {
    final buffer = StringBuffer();
    for (final msg in messages) {
      // Gemma uses "model" for the assistant role.
      final role = msg.role == ChatRole.assistant ? 'model' : msg.role.name;
      buffer.write('<start_of_turn>$role\n');
      buffer.write('${msg.content}<end_of_turn>\n');
    }
    buffer.write('<start_of_turn>model\n');
    return buffer.toString();
  }
}
