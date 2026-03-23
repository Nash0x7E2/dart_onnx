import 'dart:convert';
import 'dart:io';

import '../config/tokenizer_config.dart';

/// A Byte-Level BPE tokenizer compatible with Hugging Face `tokenizer.json`.
///
/// Encodes text into token IDs and decodes token IDs back to text.
///
/// ```dart
/// final tokenizer = await Tokenizer.fromFile('path/to/tokenizer.json');
/// final ids = tokenizer.encode('Hello, world!');
/// final text = tokenizer.decode(ids);
/// ```
class Tokenizer {
  /// The vocabulary: maps token string → token ID.
  final Map<String, int> _vocab;

  /// The reverse vocabulary: maps token ID → token string.
  final Map<int, String> _reverseVocab;

  /// Ordered list of BPE merge pairs.
  final List<(String, String)> _merges;

  /// Set of special token strings (e.g., `<s>`, `</s>`).
  final Set<String> _specialTokens;

  /// Optional tokenizer config for chat templates.
  final TokenizerConfig? tokenizerConfig;

  Tokenizer._({
    required Map<String, int> vocab,
    required Map<int, String> reverseVocab,
    required List<(String, String)> merges,
    required Set<String> specialTokens,
    this.tokenizerConfig,
  }) : _vocab = vocab,
       _reverseVocab = reverseVocab,
       _merges = merges,
       _specialTokens = specialTokens;

  /// Loads a [Tokenizer] from a Hugging Face `tokenizer.json` file.
  ///
  /// Optionally accepts a [TokenizerConfig] for chat template support.
  static Future<Tokenizer> fromFile(
    String path, {
    TokenizerConfig? config,
  }) async {
    final content = await File(path).readAsString();
    final json = jsonDecode(content) as Map<String, dynamic>;
    return Tokenizer.fromJson(json, config: config);
  }

  /// Parses a [Tokenizer] from a decoded `tokenizer.json` map.
  factory Tokenizer.fromJson(
    Map<String, dynamic> json, {
    TokenizerConfig? config,
  }) {
    // Parse vocabulary from model.vocab
    final model = json['model'] as Map<String, dynamic>;
    final vocabJson = model['vocab'] as Map<String, dynamic>;
    final vocab = vocabJson.map((k, v) => MapEntry(k, v as int));
    final reverseVocab = vocab.map((k, v) => MapEntry(v, k));

    // Parse BPE merges
    final mergesJson = model['merges'] as List<dynamic>;
    final merges = <(String, String)>[];
    for (final merge in mergesJson) {
      final parts = (merge as String).split(' ');
      if (parts.length == 2) {
        merges.add((parts[0], parts[1]));
      }
    }

    // Parse added_tokens for special tokens
    final addedTokens = json['added_tokens'] as List<dynamic>?;
    final specialTokens = <String>{};
    if (addedTokens != null) {
      for (final token in addedTokens) {
        final tokenMap = token as Map<String, dynamic>;
        if (tokenMap['special'] == true) {
          final content = tokenMap['content'] as String;
          specialTokens.add(content);
          // Ensure special tokens are in the vocabulary.
          final id = tokenMap['id'] as int;
          vocab[content] = id;
          reverseVocab[id] = content;
        }
      }
    }

    return Tokenizer._(
      vocab: vocab,
      reverseVocab: reverseVocab,
      merges: merges,
      specialTokens: specialTokens,
      tokenizerConfig: config,
    );
  }

  /// The size of the vocabulary.
  int get vocabSize => _vocab.length;

  /// Returns the token ID for a given token string, or null if not found.
  int? tokenToId(String token) => _vocab[token];

  /// Returns the token string for a given token ID, or null if not found.
  String? idToToken(int id) => _reverseVocab[id];

  /// Encodes a [text] string into a list of token IDs.
  ///
  /// This implements byte-level BPE tokenization:
  /// 1. Converts the text into byte-level characters.
  /// 2. Iteratively merges the most preferred pairs according to the merge list.
  /// 3. Looks up each resulting token in the vocabulary.
  List<int> encode(String text) {
    if (text.isEmpty) return [];

    // Split the text into pre-tokenized chunks, preserving special tokens.
    final chunks = _preTokenize(text);
    final result = <int>[];

    for (final chunk in chunks) {
      // Check if this chunk is a special token.
      if (_specialTokens.contains(chunk)) {
        final id = _vocab[chunk];
        if (id != null) result.add(id);
        continue;
      }

      // Convert to byte-level representation.
      final byteTokens = _textToByteTokens(chunk);

      // Apply BPE merges.
      final merged = _applyBpeMerges(byteTokens);

      // Look up each token in the vocabulary.
      for (final token in merged) {
        final id = _vocab[token];
        if (id != null) {
          result.add(id);
        }
        // Unknown tokens are silently dropped for now.
        // A more robust implementation would use an <unk> fallback.
      }
    }

    return result;
  }

  /// Decodes a list of token [ids] back into a text string.
  String decode(List<int> ids) {
    final buffer = StringBuffer();
    for (final id in ids) {
      final token = _reverseVocab[id];
      if (token != null && !_specialTokens.contains(token)) {
        buffer.write(_byteTokenToText(token));
      }
    }
    return buffer.toString();
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  /// Pre-tokenizes text by splitting out special tokens as separate chunks.
  List<String> _preTokenize(String text) {
    if (_specialTokens.isEmpty) return [text];

    final chunks = <String>[];
    var remaining = text;

    while (remaining.isNotEmpty) {
      // Find the earliest special token occurrence.
      int? earliestIndex;
      String? earliestToken;
      for (final special in _specialTokens) {
        final index = remaining.indexOf(special);
        if (index != -1 && (earliestIndex == null || index < earliestIndex)) {
          earliestIndex = index;
          earliestToken = special;
        }
      }

      if (earliestIndex == null) {
        // No more special tokens found.
        chunks.add(remaining);
        break;
      }

      // Add the text before the special token.
      if (earliestIndex > 0) {
        chunks.add(remaining.substring(0, earliestIndex));
      }
      // Add the special token itself.
      chunks.add(earliestToken!);
      remaining = remaining.substring(earliestIndex + earliestToken.length);
    }

    return chunks;
  }

  /// Converts a text chunk to byte-level token representation.
  ///
  /// Uses the GPT-2 byte-to-unicode mapping.
  List<String> _textToByteTokens(String text) {
    final bytes = utf8.encode(text);
    return bytes.map((b) => _byteToUnicode[b] ?? '?').toList();
  }

  /// Converts a byte-level token string back to text.
  String _byteTokenToText(String token) {
    final bytes = <int>[];
    for (final char in token.runes) {
      final byte = _unicodeToByte[char];
      if (byte != null) {
        bytes.add(byte);
      }
    }
    try {
      return utf8.decode(bytes, allowMalformed: true);
    } catch (_) {
      return token;
    }
  }

  /// Applies BPE merges to a list of tokens iteratively.
  List<String> _applyBpeMerges(List<String> tokens) {
    if (tokens.length <= 1) return tokens;

    var word = List<String>.from(tokens);

    for (final (first, second) in _merges) {
      var i = 0;
      while (i < word.length - 1) {
        if (word[i] == first && word[i + 1] == second) {
          word = [
            ...word.sublist(0, i),
            '$first$second',
            ...word.sublist(i + 2),
          ];
        } else {
          i++;
        }
      }
      if (word.length == 1) break;
    }

    return word;
  }

  // ---------------------------------------------------------------------------
  // GPT-2 byte-to-unicode mapping
  // ---------------------------------------------------------------------------

  /// Lazily computed GPT-2 byte ↔ unicode mapping.
  static final Map<int, String> _byteToUnicode = _buildByteToUnicode();
  static final Map<int, int> _unicodeToByte = _buildUnicodeToByte();

  static Map<int, String> _buildByteToUnicode() {
    // This is the standard GPT-2 bytes_to_unicode() mapping.
    final bs = <int>[];
    final cs = <int>[];

    // Printable ASCII ranges.
    for (var b = '!'.codeUnitAt(0); b <= '~'.codeUnitAt(0); b++) {
      bs.add(b);
      cs.add(b);
    }
    for (var b = '¡'.codeUnitAt(0); b <= '¬'.codeUnitAt(0); b++) {
      bs.add(b);
      cs.add(b);
    }
    for (var b = '®'.codeUnitAt(0); b <= 'ÿ'.codeUnitAt(0); b++) {
      bs.add(b);
      cs.add(b);
    }

    // Map remaining bytes to higher unicode code points.
    var n = 0;
    for (var b = 0; b < 256; b++) {
      if (!bs.contains(b)) {
        bs.add(b);
        cs.add(256 + n);
        n++;
      }
    }

    final map = <int, String>{};
    for (var i = 0; i < bs.length; i++) {
      map[bs[i]] = String.fromCharCode(cs[i]);
    }
    return map;
  }

  static Map<int, int> _buildUnicodeToByte() {
    final map = <int, int>{};
    for (final entry in _byteToUnicode.entries) {
      map[entry.value.codeUnitAt(0)] = entry.key;
    }
    return map;
  }
}
