import 'dart:convert';
import 'dart:io';

/// Configuration parsed from a Hugging Face `config.json` file.
///
/// Contains the structural parameters of the model: vocabulary size,
/// number of layers, attention head counts, etc.
///
/// Supports both flat configs (SmolLM2, Llama) and nested configs where
/// model parameters live under a `text_config` key (Gemma 4).
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

  /// The dimension of full-attention heads (for models with mixed attention).
  ///
  /// Models like Gemma 4 use sliding attention for most layers but full
  /// attention every N layers. The full-attention layers use this larger
  /// head dimension. Null if the model uses uniform attention.
  final int? globalHeadDim;

  /// Per-layer attention type identifiers (e.g. `sliding_attention`,
  /// `full_attention`).
  ///
  /// Used by models like Gemma 4 that mix attention types across layers.
  /// Null if the model uses uniform attention.
  final List<String>? layerTypes;

  /// The maximum sequence length the model supports.
  final int maxPositionEmbeddings;

  /// The model type identifier (e.g., "llama", "gemma3").
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
    this.globalHeadDim,
    this.layerTypes,
    required this.maxPositionEmbeddings,
    required this.modelType,
    this.bosTokenId,
    this.eosTokenId,
    this.padTokenId,
    required this.raw,
  });

  /// Parses a [ModelConfig] from a decoded JSON map.
  ///
  /// Automatically detects whether model parameters are at the root level
  /// (SmolLM2, Llama) or nested under `text_config` (Gemma 4).
  factory ModelConfig.fromJson(Map<String, dynamic> json) {
    // Some models (e.g. Gemma 4) nest model params under text_config.
    final cfg = json['text_config'] as Map<String, dynamic>? ?? json;

    final hiddenSize = cfg['hidden_size'] as int;
    final numAttentionHeads = cfg['num_attention_heads'] as int;

    // eos_token_id can be int, list of ints, or null.
    final rawEos = json['eos_token_id'];
    int? eosTokenId;
    if (rawEos is int) {
      eosTokenId = rawEos;
    } else if (rawEos is List && rawEos.isNotEmpty) {
      eosTokenId = rawEos.first as int;
    }

    // Parse layer types if present (e.g. Gemma 4 mixed attention).
    List<String>? layerTypes;
    final rawLayerTypes = cfg['layer_types'];
    if (rawLayerTypes is List) {
      layerTypes = rawLayerTypes.cast<String>();
    }

    return ModelConfig(
      vocabSize: cfg['vocab_size'] as int,
      numHiddenLayers: cfg['num_hidden_layers'] as int,
      numAttentionHeads: numAttentionHeads,
      numKeyValueHeads:
          (cfg['num_key_value_heads'] as int?) ?? numAttentionHeads,
      hiddenSize: hiddenSize,
      intermediateSize: cfg['intermediate_size'] as int,
      headDim: (cfg['head_dim'] as int?) ?? (hiddenSize ~/ numAttentionHeads),
      globalHeadDim: cfg['global_head_dim'] as int?,
      layerTypes: layerTypes,
      maxPositionEmbeddings: cfg['max_position_embeddings'] as int,
      modelType: json['model_type'] as String,
      bosTokenId: json['bos_token_id'] as int?,
      eosTokenId: eosTokenId,
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
