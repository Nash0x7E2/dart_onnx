import 'dart:convert';
import 'dart:io';

import '../config/tokenizer_config.dart';

/// A Byte-Level BPE tokenizer compatible with Hugging Face tokenizer.json.
///
/// Implements the full pre-tokenization pipeline used by GPT-2/SmolLM2/Llama:
///   1. Split out special tokens.
///   2. Apply the GPT-2 regex to split on word boundaries.
///   3. Split individual digits (Digits pre-tokenizer).
///   4. Convert each word to byte-level characters.
///   5. Apply BPE merges.
///   6. Look up token IDs.
class Tokenizer {
  /// The vocabulary: maps token string to token ID.
  final Map<String, int> _vocab;

  /// The reverse vocabulary: maps token ID to token string.
  final Map<int, String> _reverseVocab;

  /// Merge rank lookup: maps "first second" to priority index.
  final Map<String, int> _mergeRanks;

  /// Set of special token strings.
  final Set<String> _specialTokens;

  /// Whether to split individual digits before byte-level encoding.
  final bool _splitDigits;

  /// Optional tokenizer config for chat templates.
  final TokenizerConfig? tokenizerConfig;

  /// The GPT-2 pre-tokenization regex.
  ///
  /// This is the exact regex used by the HuggingFace ByteLevel pre-tokenizer
  /// (from tokenizers/src/pre_tokenizers/byte_level.rs).
  ///
  /// It splits text into: contractions, optional-space+letters,
  /// optional-space+numbers, optional-space+punctuation, trailing whitespace
  /// (collapsed), and individual whitespace chars.
  static final _gpt2Regex = RegExp(
    r"'(?:[sdmt]|ll|ve|re)"
    r"| ?\p{L}+"
    r"| ?\p{N}+"
    r"| ?[^\s\p{L}\p{N}]+"
    r"|\s+(?!\S)"
    r"|\s+",
    unicode: true,
  );

  Tokenizer._({
    required Map<String, int> vocab,
    required Map<int, String> reverseVocab,
    required Map<String, int> mergeRanks,
    required Set<String> specialTokens,
    required bool splitDigits,
    this.tokenizerConfig,
  })  : _vocab = vocab,
        _reverseVocab = reverseVocab,
        _mergeRanks = mergeRanks,
        _specialTokens = specialTokens,
        _splitDigits = splitDigits;

  /// Loads a [Tokenizer] from a Hugging Face tokenizer.json file.
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

  /// Parses a [Tokenizer] from a decoded tokenizer.json map.
  factory Tokenizer.fromJson(
    Map<String, dynamic> json, {
    TokenizerConfig? config,
  }) {
    // Parse vocabulary from model.vocab
    final model = json['model'] as Map<String, dynamic>;
    final vocabJson = model['vocab'] as Map<String, dynamic>;
    final vocab = vocabJson.map((k, v) => MapEntry(k, v as int));
    final reverseVocab = vocab.map((k, v) => MapEntry(v, k));

    // Parse BPE merges and build rank lookup
    final mergesJson = model['merges'] as List<dynamic>;
    final mergeRanks = <String, int>{};
    for (var i = 0; i < mergesJson.length; i++) {
      final entry = mergesJson[i];
      String first;
      String second;

      if (entry is List) {
        // Format: [["a", "b"], ...] (used by SmolLM2 and newer tokenizers).
        first = entry[0] as String;
        second = entry[1] as String;
      } else if (entry is String) {
        // Format: ["a b", ...] (space-separated, used by older tokenizers).
        final parts = entry.split(' ');
        if (parts.length != 2) continue;
        first = parts[0];
        second = parts[1];
      } else {
        continue;
      }

      mergeRanks['$first $second'] = i;
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

    // Detect pre-tokenizer configuration.
    var splitDigits = false;
    final preTokenizer = json['pre_tokenizer'] as Map<String, dynamic>?;
    if (preTokenizer != null) {
      final type = preTokenizer['type'] as String?;
      if (type == 'Digits') {
        splitDigits = preTokenizer['individual_digits'] == true;
      } else if (type == 'Sequence') {
        final pretokenizers =
            preTokenizer['pretokenizers'] as List<dynamic>? ?? [];
        for (final pt in pretokenizers) {
          final ptMap = pt as Map<String, dynamic>;
          if (ptMap['type'] == 'Digits' &&
              ptMap['individual_digits'] == true) {
            splitDigits = true;
          }
        }
      }
    }

    return Tokenizer._(
      vocab: vocab,
      reverseVocab: reverseVocab,
      mergeRanks: mergeRanks,
      specialTokens: specialTokens,
      splitDigits: splitDigits,
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
  /// 1. Splits out special tokens.
  /// 2. Applies the GPT-2 regex to split on word boundaries.
  /// 3. Optionally splits individual digits.
  /// 4. Converts each piece to byte-level characters.
  /// 5. Iteratively merges the best pairs according to the merge list.
  /// 6. Looks up each resulting token in the vocabulary.
  List<int> encode(String text) {
    if (text.isEmpty) return [];

    // Step 1: Split out special tokens.
    final segments = _splitSpecialTokens(text);
    final result = <int>[];

    for (final segment in segments) {
      // Check if this segment is a special token.
      if (_specialTokens.contains(segment)) {
        final id = _vocab[segment];
        if (id != null) result.add(id);
        continue;
      }

      // Step 2: Apply GPT-2 regex pre-tokenization.
      final words = _regexPreTokenize(segment);

      for (final word in words) {
        // Step 3: Optionally split digits.
        final pieces = _splitDigits ? _splitIndividualDigits(word) : [word];

        for (final piece in pieces) {
          // Step 4: Convert to byte-level representation.
          final byteTokens = _textToByteTokens(piece);

          // Step 5: Apply BPE merges.
          final merged = _applyBpeMerges(byteTokens);

          // Step 6: Look up each token in the vocabulary.
          for (final token in merged) {
            final id = _vocab[token];
            if (id != null) {
              result.add(id);
            }
            // Unknown tokens are silently dropped for now.
          }
        }
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
  // Pre-tokenization helpers
  // ---------------------------------------------------------------------------

  /// Splits text into segments, isolating special tokens as separate chunks.
  List<String> _splitSpecialTokens(String text) {
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

  /// Applies the GPT-2 regex to split text into pre-tokenized words.
  ///
  /// Each match becomes a separate "word" that will be independently
  /// byte-encoded and BPE-merged. This is critical for matching the
  /// reference Python tokenizer output.
  List<String> _regexPreTokenize(String text) {
    final matches = _gpt2Regex.allMatches(text);
    return matches.map((m) => m.group(0)!).toList();
  }

  /// Splits a string so that each ASCII digit (0-9) is its own element.
  ///
  /// Non-digit characters are kept grouped together. This matches the
  /// HuggingFace Digits(individual_digits=true) pre-tokenizer.
  List<String> _splitIndividualDigits(String text) {
    final result = <String>[];
    final buffer = StringBuffer();

    for (var i = 0; i < text.length; i++) {
      final char = text[i];
      if (char.codeUnitAt(0) >= 0x30 && char.codeUnitAt(0) <= 0x39) {
        // It's a digit. Flush any buffered non-digit text first.
        if (buffer.isNotEmpty) {
          result.add(buffer.toString());
          buffer.clear();
        }
        result.add(char);
      } else {
        buffer.write(char);
      }
    }

    if (buffer.isNotEmpty) {
      result.add(buffer.toString());
    }

    return result;
  }

  // ---------------------------------------------------------------------------
  // Byte-level encoding/decoding
  // ---------------------------------------------------------------------------

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

  // ---------------------------------------------------------------------------
  // BPE merges
  // ---------------------------------------------------------------------------

  /// Applies BPE merges to a list of tokens using merge ranks for efficiency.
  ///
  /// Instead of iterating through all merges linearly, this implementation
  /// finds the highest-priority (lowest rank) pair in the current word and
  /// merges it, repeating until no more merges apply.
  List<String> _applyBpeMerges(List<String> tokens) {
    if (tokens.length <= 1) return tokens;

    var word = List<String>.from(tokens);

    while (true) {
      // Find the pair with the lowest merge rank.
      int? bestRank;
      int? bestIndex;

      for (var i = 0; i < word.length - 1; i++) {
        final key = '${word[i]} ${word[i + 1]}';
        final rank = _mergeRanks[key];
        if (rank != null && (bestRank == null || rank < bestRank)) {
          bestRank = rank;
          bestIndex = i;
        }
      }

      if (bestIndex == null) break; // No more merges.

      // Merge the pair at bestIndex.
      final merged = '${word[bestIndex]}${word[bestIndex + 1]}';
      word = [
        ...word.sublist(0, bestIndex),
        merged,
        ...word.sublist(bestIndex + 2),
      ];

      if (word.length == 1) break;
    }

    return word;
  }

  // ---------------------------------------------------------------------------
  // GPT-2 byte-to-unicode mapping
  // ---------------------------------------------------------------------------

  /// Lazily computed GPT-2 byte <-> unicode mapping.
  static final Map<int, String> _byteToUnicode = _buildByteToUnicode();
  static final Map<int, int> _unicodeToByte = _buildUnicodeToByte();

  static Map<int, String> _buildByteToUnicode() {
    // This is the standard GPT-2 bytes_to_unicode() mapping.
    final bs = <int>[];
    final cs = <int>[];

    // Printable ASCII ranges.
    for (var b = 0x21; b <= 0x7E; b++) {
      bs.add(b);
      cs.add(b);
    }
    for (var b = 0xA1; b <= 0xAC; b++) {
      bs.add(b);
      cs.add(b);
    }
    for (var b = 0xAE; b <= 0xFF; b++) {
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
