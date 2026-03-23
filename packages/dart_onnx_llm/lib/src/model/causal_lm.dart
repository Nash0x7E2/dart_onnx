import 'dart:typed_data';

import 'package:dart_onnx/dart_onnx.dart';

import '../config/generation_config.dart';
import '../config/model_config.dart';
import '../sampler/sampler.dart';

/// A wrapper around [DartONNXSession] for causal (autoregressive) language
/// models.
///
/// Handles input tensor construction, KV cache management, and provides
/// a streaming generation interface.
///
/// ## What is CausalLM?
///
/// "CausalLM" stands for *Causal Language Modeling*. It describes autoregressive
/// models (like GPT, Llama, SmolLM) that predict the next token based strictly
/// on preceding context. This is distinct from masked language models (like
/// BERT) which can attend to tokens in both directions.
class CausalLM {
  final DartONNXSession _session;
  final ModelConfig config;

  CausalLM._({
    required DartONNXSession session,
    required this.config,
  }) : _session = session;

  /// Loads a [CausalLM] from a `.onnx` model file and a [ModelConfig].
  factory CausalLM.fromFile(
    String modelPath, {
    required ModelConfig config,
    DartONNXLoggingLevel loggingLevel = DartONNXLoggingLevel.warning,
    List<DartONNXExecutionProvider> executionProviders = const [
      DartONNXExecutionProvider.cpu,
    ],
  }) {
    final session = DartONNXSession.fromFile(
      DartONNX(loggingLevel: loggingLevel),
      modelPath,
      executionProviders: executionProviders,
    );

    return CausalLM._(session: session, config: config);
  }

  /// The names of the model's input tensors.
  List<String> get inputNames => _session.inputNames;

  /// The names of the model's output tensors.
  List<String> get outputNames => _session.outputNames;

  /// Whether the model expects past_key_values inputs (KV cache).
  bool get usesKvCache =>
      _session.inputNames.any((n) => n.startsWith('past_key_values.'));

  /// Generates a stream of token IDs autoregressively from [inputIds].
  ///
  /// The generation loop runs until [GenerationConfig.maxTokens] is reached,
  /// or an EOS token is generated.
  Stream<int> generate(
    List<int> inputIds, {
    GenerationConfig config = const GenerationConfig(),
    Sampler? sampler,
  }) async* {
    sampler ??= Sampler(config);

    final stopTokenIds = config.stopTokenIds ??
        (this.config.eosTokenId != null ? [this.config.eosTokenId!] : <int>[]);

    final generatedTokenIds = <int>[];

    // --- State for KV cache ---
    var currentInputIds = List<int>.from(inputIds);
    var pastSequenceLength = 0;
    Map<String, DartONNXTensor>? kvCache;

    for (var step = 0; step < config.maxTokens; step++) {
      final seqLen = currentInputIds.length;
      const batch = 1;

      // Build input tensors
      final inputs = <String, DartONNXTensor>{};

      // input_ids [1, seqLen]
      inputs['input_ids'] = DartONNXTensor.int64(
        data: Int64List.fromList(currentInputIds),
        shape: [batch, seqLen],
      );

      // attention_mask [1, pastSeqLen + seqLen] — all 1s
      final totalSeqLen = pastSequenceLength + seqLen;
      inputs['attention_mask'] = DartONNXTensor.int64(
        data: Int64List.fromList(List.filled(totalSeqLen, 1)),
        shape: [batch, totalSeqLen],
      );

      // position_ids [1, seqLen]
      if (_session.inputNames.contains('position_ids')) {
        inputs['position_ids'] = DartONNXTensor.int64(
          data: Int64List.fromList(
            List.generate(seqLen, (i) => pastSequenceLength + i),
          ),
          shape: [batch, seqLen],
        );
      }

      // past_key_values — either from previous step or zeroed out
      final kvInputNames = _session.inputNames
          .where((n) => n.startsWith('past_key_values.'))
          .toList();

      if (kvInputNames.isNotEmpty) {
        if (kvCache != null) {
          // Reuse tensors from previous step's outputs.
          for (final name in kvInputNames) {
            inputs[name] = kvCache[name]!;
          }
        } else {
          // Initial step: provide empty (zero) past_key_values.
          for (final name in kvInputNames) {
            inputs[name] = DartONNXTensor.float32(
              data: Float32List(0),
              shape: [
                batch,
                this.config.numKeyValueHeads,
                0,
                this.config.headDim,
              ],
            );
          }
        }
      }

      // Run the model
      final outputs = _session.run(inputs);

      // Extract logits — shape [1, seqLen, vocabSize]
      final logitsTensor = outputs['logits']!;
      final logitsData = logitsTensor.data as Float32List;
      final vocabSize = this.config.vocabSize;

      // Get the logits for the last position.
      final lastTokenOffset = (seqLen - 1) * vocabSize;
      final lastTokenLogits = Float32List.fromList(
        logitsData.sublist(lastTokenOffset, lastTokenOffset + vocabSize),
      );

      // Sample the next token.
      final nextTokenId = sampler.sample(
        lastTokenLogits,
        generatedTokenIds: generatedTokenIds,
      );

      // Dispose input tensors (except reused KV cache ones).
      inputs['input_ids']!.dispose();
      inputs['attention_mask']!.dispose();
      if (inputs.containsKey('position_ids')) {
        inputs['position_ids']!.dispose();
      }

      // Dispose old KV cache tensors (from the *previous* step).
      if (kvCache != null) {
        for (final tensor in kvCache.values) {
          tensor.dispose();
        }
      }

      // Extract new KV cache from outputs for the next step.
      kvCache = <String, DartONNXTensor>{};
      for (final name in outputs.keys) {
        if (name.startsWith('present')) {
          // Map "present.X.key" → "past_key_values.X.key"
          final pastName = name.replaceFirst('present', 'past_key_values');
          kvCache[pastName] = outputs[name]!;
        }
      }

      // Dispose logits tensor (we've already extracted the data).
      logitsTensor.dispose();

      // Update state for next iteration.
      pastSequenceLength += seqLen;
      currentInputIds = [nextTokenId];
      generatedTokenIds.add(nextTokenId);

      // Check for stop condition.
      if (stopTokenIds.contains(nextTokenId)) break;

      yield nextTokenId;
    }

    // Dispose final KV cache.
    if (kvCache != null) {
      for (final tensor in kvCache.values) {
        tensor.dispose();
      }
    }
  }

  /// Disposes the underlying ONNX session and environment.
  void dispose() {
    _session.dispose();
  }
}
