import 'dart:convert';
import 'dart:io';

/// Configuration parsed from a Hugging Face `config.json` file.
///
/// Contains the structural parameters of the model: vocabulary size,
/// number of layers, attention head counts, etc.
class ModelConfig {
  /// The size of the model's vocabulary.
  final int vocabSize;

  /// The number of hidden (transformer) layers.
  final int numHiddenLayers;

  /// The number of attention heads.
  final int numAttentionHeads;

  /// The number of key/value attention heads (for grouped-query attention).
  final int numKeyValueHeads;

  /// The hidden size (embedding dimension).
  final int hiddenSize;

  /// The intermediate (feed-forward) size.
  final int intermediateSize;

  /// The dimension of each attention head.
  ///
  /// Typically `hiddenSize ~/ numAttentionHeads`.
  final int headDim;

  /// The maximum sequence length the model supports.
  final int maxPositionEmbeddings;

  /// The model type identifier (e.g., "llama", "gpt2").
  final String modelType;

  /// The token ID for beginning-of-sequence, if any.
  final int? bosTokenId;

  /// The token ID for end-of-sequence, if any.
  final int? eosTokenId;

  /// The token ID used for padding, if any.
  final int? padTokenId;

  /// The raw JSON map for accessing any additional fields.
  final Map<String, dynamic> raw;

  const ModelConfig({
    required this.vocabSize,
    required this.numHiddenLayers,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    required this.hiddenSize,
    required this.intermediateSize,
    required this.headDim,
    required this.maxPositionEmbeddings,
    required this.modelType,
    this.bosTokenId,
    this.eosTokenId,
    this.padTokenId,
    required this.raw,
  });

  /// Parses a [ModelConfig] from a decoded JSON map.
  factory ModelConfig.fromJson(Map<String, dynamic> json) {
    final hiddenSize = json['hidden_size'] as int;
    final numAttentionHeads = json['num_attention_heads'] as int;

    return ModelConfig(
      vocabSize: json['vocab_size'] as int,
      numHiddenLayers: json['num_hidden_layers'] as int,
      numAttentionHeads: numAttentionHeads,
      numKeyValueHeads:
          (json['num_key_value_heads'] as int?) ?? numAttentionHeads,
      hiddenSize: hiddenSize,
      intermediateSize: json['intermediate_size'] as int,
      headDim: (json['head_dim'] as int?) ?? (hiddenSize ~/ numAttentionHeads),
      maxPositionEmbeddings: json['max_position_embeddings'] as int,
      modelType: json['model_type'] as String,
      bosTokenId: json['bos_token_id'] as int?,
      eosTokenId: json['eos_token_id'] as int?,
      padTokenId: json['pad_token_id'] as int?,
      raw: json,
    );
  }

  /// Loads a [ModelConfig] from a `config.json` file.
  static Future<ModelConfig> fromFile(String path) async {
    final content = await File(path).readAsString();
    final json = jsonDecode(content) as Map<String, dynamic>;
    return ModelConfig.fromJson(json);
  }
}
