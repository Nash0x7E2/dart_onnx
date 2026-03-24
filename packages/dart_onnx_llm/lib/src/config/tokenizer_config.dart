import 'dart:convert';
import 'dart:io';

/// Configuration parsed from a Hugging Face `tokenizer_config.json` file.
///
/// Contains metadata about the tokenizer: special tokens, chat templates, etc.
class TokenizerConfig {
  /// The chat template string (Jinja2 format) used to format conversations.
  final String? chatTemplate;

  /// The beginning-of-sequence token string (e.g., `<s>`).
  final String? bosToken;

  /// The end-of-sequence token string (e.g., `</s>`).
  final String? eosToken;

  /// The padding token string (e.g., `<pad>`).
  final String? padToken;

  /// The unknown token string (e.g., `<unk>`).
  final String? unkToken;

  /// Map of all special tokens by name.
  final Map<String, String> specialTokens;

  /// The raw JSON map for accessing any additional fields.
  final Map<String, dynamic> raw;

  const TokenizerConfig({
    this.chatTemplate,
    this.bosToken,
    this.eosToken,
    this.padToken,
    this.unkToken,
    required this.specialTokens,
    required this.raw,
  });

  /// Parses a [TokenizerConfig] from a decoded JSON map.
  factory TokenizerConfig.fromJson(Map<String, dynamic> json) {
    // Special tokens can be either a plain string or a map with a "content" key.
    String? extractToken(dynamic value) {
      if (value == null) return null;
      if (value is String) return value;
      if (value is Map<String, dynamic>) return value['content'] as String?;
      return null;
    }

    final specialTokens = <String, String>{};

    // Collect any added_tokens_decoder entries that are marked as special.
    final addedTokensDecoder =
        json['added_tokens_decoder'] as Map<String, dynamic>?;
    if (addedTokensDecoder != null) {
      for (final entry in addedTokensDecoder.entries) {
        final tokenData = entry.value;
        if (tokenData is Map<String, dynamic> && tokenData['special'] == true) {
          final content = tokenData['content'] as String?;
          if (content != null) {
            specialTokens[content] = content;
          }
        }
      }
    }

    return TokenizerConfig(
      chatTemplate: json['chat_template'] as String?,
      bosToken: extractToken(json['bos_token']),
      eosToken: extractToken(json['eos_token']),
      padToken: extractToken(json['pad_token']),
      unkToken: extractToken(json['unk_token']),
      specialTokens: specialTokens,
      raw: json,
    );
  }

  /// Loads a [TokenizerConfig] from a `tokenizer_config.json` file.
  static Future<TokenizerConfig> fromFile(String path) async {
    final content = await File(path).readAsString();
    final json = jsonDecode(content) as Map<String, dynamic>;
    return TokenizerConfig.fromJson(json);
  }
}
