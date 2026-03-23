/// Represents a single message in a chat conversation.
///
/// Messages have a [role] (system, user, or assistant) and [content].
class ChatMessage {
  /// The role of the message sender.
  final ChatRole role;

  /// The text content of the message.
  ///
  /// For multimodal messages in the future, this will be extended to
  /// support [MessagePart] lists alongside plain text.
  final String content;

  const ChatMessage({required this.role, required this.content});

  /// Creates a system message.
  const ChatMessage.system(this.content) : role = ChatRole.system;

  /// Creates a user message.
  const ChatMessage.user(this.content) : role = ChatRole.user;

  /// Creates an assistant message.
  const ChatMessage.assistant(this.content) : role = ChatRole.assistant;

  @override
  String toString() => 'ChatMessage(${role.name}: $content)';
}

/// The role of a participant in a chat conversation.
enum ChatRole {
  /// A system instruction that shapes the assistant's behavior.
  system,

  /// A message from the end user.
  user,

  /// A message from the AI assistant.
  assistant,
}
