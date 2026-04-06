import 'dart:typed_data';

import 'package:dart_onnx/dart_onnx.dart';

import '../config/generation_config.dart';
import '../config/model_config.dart';
import '../sampler/sampler.dart';

/// A wrapper around ONNX sessions for causal (autoregressive) language models.
///
/// Handles input tensor construction, KV cache management, and provides
/// a streaming generation interface.
///
/// Supports monolithic models (e.g. Llama, SmolLM2) and split-graph models
/// (e.g. Gemma 4) where embeddings and decoders are in separate sessions.
class CausalLM {
  /// The global Environment instance.
  ///
  /// CRITICAL: We must maintain a hard reference to this object. If the
  /// environment is garbage collected, Dart's NativeFinalizer will destroy
  /// the underlying OrtEnv, causing a Segmentation Fault if sessions are
  /// still using it!
  final DartONNX _env;

  /// The main decoder session (or the monolithic model session).
  final DartONNXSession _decoderSession;

  /// The optional embedding session (for split-graph architectures).
  final DartONNXSession? _embedSession;

  final ModelConfig config;

  CausalLM._({
    required DartONNX env,
    required DartONNXSession decoderSession,
    DartONNXSession? embedSession,
    required this.config,
  })  : _env = env,
        _decoderSession = decoderSession,
        _embedSession = embedSession;

  /// Loads a [CausalLM] from a single monolithic `.onnx` file.
  factory CausalLM.fromFile(
    String modelPath, {
    required ModelConfig config,
    DartONNXLoggingLevel loggingLevel = DartONNXLoggingLevel.warning,
    List<DartONNXExecutionProvider> executionProviders = const [
      DartONNXExecutionProvider.cpu,
    ],
  }) {
    final env = DartONNX(loggingLevel: loggingLevel);
    final session = DartONNXSession.fromFile(
      env,
      modelPath,
      executionProviders: executionProviders,
    );

    return CausalLM._(
      env: env,
      decoderSession: session,
      config: config,
    );
  }

  /// Loads a [CausalLM] from a split-graph system (separate embed and decoder).
  ///
  /// The [embedPath] session takes token IDs and outputs `inputs_embeds` and
  /// `per_layer_inputs`. The [decoderPath] session takes those and outputs logits.
  factory CausalLM.fromPipeline(
    String decoderPath, {
    required String embedPath,
    required ModelConfig config,
    DartONNXLoggingLevel loggingLevel = DartONNXLoggingLevel.warning,
    List<DartONNXExecutionProvider> executionProviders = const [
      DartONNXExecutionProvider.cpu,
    ],
  }) {
    // Both sessions MUST share the same environment to safely pass tensors
    // between them without memory corruption.
    final env = DartONNX(loggingLevel: loggingLevel);

    final embedSession = DartONNXSession.fromFile(
      env,
      embedPath,
      executionProviders: executionProviders,
    );

    final decoderSession = DartONNXSession.fromFile(
      env,
      decoderPath,
      executionProviders: executionProviders,
    );

    return CausalLM._(
      env: env,
      decoderSession: decoderSession,
      embedSession: embedSession,
      config: config,
    );
  }

  /// Returns true if this model uses a split-graph architecture.
  bool get isSplitGraph => _embedSession != null;

  /// Generates a stream of token IDs autoregressively from [inputIds].
  Stream<int> generate(
    List<int> inputIds, {
    GenerationConfig config = const GenerationConfig(),
    Sampler? sampler,
  }) async* {
    sampler ??= Sampler(config);

    final stopTokenIds =
        config.stopTokenIds ??
        (this.config.eosTokenId != null ? [this.config.eosTokenId!] : <int>[]);

    final contextTokenIds = List<int>.from(inputIds);

    // --- State Variables ---
    var currentInputIds = List<int>.from(inputIds);
    var pastSequenceLength = 0;
    Map<String, DartONNXTensor>? kvCache;

    // For split-graph models, the pipeline passes per_layer_inputs.
    DartONNXTensor? perLayerInputsTensor;

    for (var step = 0; step < config.maxTokens; step++) {
      final seqLen = currentInputIds.length;
      const batch = 1;

      // 1. Prepare Decoder Inputs
      final decoderInputs = <String, DartONNXTensor>{};
      final tensorsToDispose = <DartONNXTensor>[];

      void trackDisposal(DartONNXTensor t) {
        tensorsToDispose.add(t);
      }

      // Attention Mask [batch, totalSeqLen]
      final totalSeqLen = pastSequenceLength + seqLen;
      final attentionMask = DartONNXTensor.int64(
        data: Int64List.fromList(List.filled(totalSeqLen, 1)),
        shape: [batch, totalSeqLen],
      );
      trackDisposal(attentionMask);
      decoderInputs['attention_mask'] = attentionMask;

      // Position IDs [batch, seqLen]
      if (_decoderSession.inputNames.contains('position_ids')) {
        final posIds = DartONNXTensor.int64(
          data: Int64List.fromList(
            List.generate(seqLen, (i) => pastSequenceLength + i),
          ),
          shape: [batch, seqLen],
        );
        trackDisposal(posIds);
        decoderInputs['position_ids'] = posIds;
      }

      // 2. Handle Split-Graph vs Monolithic
      if (isSplitGraph) {
        // Run Embedding Session First
        final embedInputs = <String, DartONNXTensor>{};
        final inputIdsTensor = DartONNXTensor.int64(
          data: Int64List.fromList(currentInputIds),
          shape: [batch, seqLen],
        );
        embedInputs['input_ids'] = inputIdsTensor;

        // Execute embedding
        final embedOutputs = _embedSession!.run(embedInputs);
        inputIdsTensor.dispose();

        // Pass embed outputs directly to decoder
        if (embedOutputs.containsKey('inputs_embeds')) {
          decoderInputs['inputs_embeds'] = embedOutputs['inputs_embeds']!;
        }

        if (embedOutputs.containsKey('per_layer_inputs')) {
          // If this is step > 0, we can reuse the previous perLayerInputsTensor.
          // Embedding only returns it on step 0, so if seqLen=1 and we have it, we keep using it.
          if (perLayerInputsTensor == null) {
            perLayerInputsTensor = embedOutputs['per_layer_inputs']!;
          } else {
             // For step > 0, the embedding session generated a new per_layer_inputs tensor.
             // We want to use the latest one because it might be different, but in Gemma 4 E2B,
             // it actually only emits it based on the input_ids dimension. Actually, Gemma 4
             // returns it every time. So just update it and we'll let NativeFinalizer handle the old one.
             perLayerInputsTensor = embedOutputs['per_layer_inputs']!;
          }
          decoderInputs['per_layer_inputs'] = perLayerInputsTensor;
        } else if (perLayerInputsTensor != null) {
          // pass the previous one
          decoderInputs['per_layer_inputs'] = perLayerInputsTensor;
        }
      } else {
        // Monolithic: Just pass input_ids to decoder
        final inputIdsTensor = DartONNXTensor.int64(
          data: Int64List.fromList(currentInputIds),
          shape: [batch, seqLen],
        );
        trackDisposal(inputIdsTensor);
        decoderInputs['input_ids'] = inputIdsTensor;
      }

      // 3. Handle KV Cache
      final kvInputNames = _decoderSession.inputNames
          .where((n) => n.startsWith('past_key_values.'))
          .toList();

      if (kvInputNames.isNotEmpty) {
        if (kvCache != null) {
          // Reuse
          for (final name in kvInputNames) {
            if (kvCache[name] != null) {
              decoderInputs[name] = kvCache[name]!;
            }
          }
        } else {
          // Initial Zeros. Support heterogeneous layer dimensions (Gemma 4).
          for (final name in kvInputNames) {
            // Extract layer index (e.g., "past_key_values.0.key" -> 0)
            final parts = name.split('.');
            final layerIdxStr = parts.length > 1 ? parts[1] : null;
            final layerIdx = layerIdxStr != null ? int.tryParse(layerIdxStr) : null;

             // Determine the correct head dimension for this layer based on attention type.
             // Gemma 4 mixes sliding_attention (head_dim) and full_attention (global_head_dim).
             int currentHeadDim = this.config.headDim;
             if (layerIdx != null &&
                 this.config.layerTypes != null &&
                 layerIdx < this.config.layerTypes!.length) {
               final attnType = this.config.layerTypes![layerIdx];
               if (attnType == 'full_attention' && this.config.globalHeadDim != null) {
                 currentHeadDim = this.config.globalHeadDim!;
               }
             }

            final zeroTensor = DartONNXTensor.float32(
              data: Float32List(0),
              shape: [
                batch,
                this.config.numKeyValueHeads,
                0, // initial sequence length
                currentHeadDim,
              ],
            );
            trackDisposal(zeroTensor);
            decoderInputs[name] = zeroTensor;
          }
        }
      }

      // 4. Handle num_logits_to_keep (Gemma 4 requires this as a scalar shape: [])
      if (_decoderSession.inputNames.contains('num_logits_to_keep')) {
         final numLogitsTensor = DartONNXTensor.int64(
           data: Int64List.fromList([1]),
           shape: [], // Empty list = Scalar
         );
         trackDisposal(numLogitsTensor);
         decoderInputs['num_logits_to_keep'] = numLogitsTensor;
      }


      // 5. Run Decoder
      final outputs = _decoderSession.run(decoderInputs);

      // Extract Logits
      final logitsTensor = outputs['logits']!;
      final logitsData = logitsTensor.data as Float32List;
      final vocabSize = this.config.vocabSize;
      
      // Determine how many tokens of logits we received.
      // E.g., if num_logits_to_keep=1, seqLen=1. If not, it could be seqLen.
      // Based on ONNX shapes, logits shape is [batch, logitsSeqLen, vocabSize].
      final logitsSeqLen = logitsData.length ~/ vocabSize;
      
      final lastTokenOffset = (logitsSeqLen - 1) * vocabSize;
      final lastTokenLogits = Float32List.fromList(
        logitsData.sublist(lastTokenOffset, lastTokenOffset + vocabSize),
      );

      // Sample
      final nextTokenId = sampler.sample(
        lastTokenLogits,
        generatedTokenIds: contextTokenIds,
      );

      // 6. Dispose Dart-created Inputs ONLY
      // Do NOT arbitrarily dispose Ort-created outputs. Rely on NativeFinalizer.
      for (final t in tensorsToDispose) {
        t.dispose();
      }

      // 7. Update State
      kvCache = <String, DartONNXTensor>{};
      for (final name in outputs.keys) {
        if (name.startsWith('present')) {
          final pastName = name.replaceFirst('present', 'past_key_values');
          kvCache[pastName] = outputs[name]!;
        }
      }

      pastSequenceLength += seqLen;
      currentInputIds = [nextTokenId];
      contextTokenIds.add(nextTokenId);

      // Check Stop Condition
      if (stopTokenIds.contains(nextTokenId)) break;

      yield nextTokenId;
    }
  }

  /// Disposes the underlying ONNX sessions and environment.
  void dispose() {
    _embedSession?.dispose();
    _decoderSession.dispose();
    _env.dispose();
  }
}
