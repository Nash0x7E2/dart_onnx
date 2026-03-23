import 'dart:math';
import 'dart:typed_data';

import '../config/generation_config.dart';

/// Processes raw logits from the model and selects the next token.
///
/// Supports greedy decoding, temperature scaling, top-k, and top-p
/// (nucleus) sampling.
class Sampler {
  final GenerationConfig _config;
  final Random _random;

  /// Creates a sampler from a [GenerationConfig].
  Sampler(this._config, {Random? random}) : _random = random ?? Random();

  /// Creates a sampler that always picks the most likely token (argmax).
  factory Sampler.greedy() => Sampler(GenerationConfig.greedy);

  /// Creates a sampler using top-p (nucleus) sampling.
  factory Sampler.topP(double p, {double temperature = 1.0}) => Sampler(
    GenerationConfig(topP: p, temperature: temperature),
  );

  /// Creates a sampler using top-k sampling.
  factory Sampler.topK(int k, {double temperature = 1.0}) => Sampler(
    GenerationConfig(topK: k, temperature: temperature),
  );

  /// Selects the next token ID from a vocabulary-sized [logits] array.
  ///
  /// Optionally accepts [generatedTokenIds] for repetition penalty.
  int sample(Float32List logits, {List<int>? generatedTokenIds}) {
    final vocabSize = logits.length;

    // 1. Apply repetition penalty
    if (_config.repetitionPenalty != 1.0 && generatedTokenIds != null) {
      for (final tokenId in generatedTokenIds) {
        if (tokenId < vocabSize) {
          if (logits[tokenId] > 0) {
            logits[tokenId] /= _config.repetitionPenalty;
          } else {
            logits[tokenId] *= _config.repetitionPenalty;
          }
        }
      }
    }

    // 2. Greedy decoding (temperature == 0)
    if (_config.temperature == 0.0) {
      return _argmax(logits);
    }

    // 3. Temperature scaling
    if (_config.temperature != 1.0) {
      for (var i = 0; i < vocabSize; i++) {
        logits[i] /= _config.temperature;
      }
    }

    // 4. Build sorted indices by logit value (descending)
    final indices = List<int>.generate(vocabSize, (i) => i)
      ..sort((a, b) => logits[b].compareTo(logits[a]));

    // 5. Apply top-k filter
    var candidates = indices;
    if (_config.topK > 0 && _config.topK < vocabSize) {
      candidates = indices.sublist(0, _config.topK);
    }

    // 6. Softmax over candidates
    final maxLogit = logits[candidates.first];
    final exps = Float64List(candidates.length);
    var sumExp = 0.0;
    for (var i = 0; i < candidates.length; i++) {
      exps[i] = exp(logits[candidates[i]] - maxLogit);
      sumExp += exps[i];
    }
    for (var i = 0; i < exps.length; i++) {
      exps[i] /= sumExp;
    }

    // 7. Apply top-p (nucleus) filtering
    if (_config.topP < 1.0) {
      var cumulative = 0.0;
      var cutoff = exps.length;
      for (var i = 0; i < exps.length; i++) {
        cumulative += exps[i];
        if (cumulative >= _config.topP) {
          cutoff = i + 1;
          break;
        }
      }
      // Re-normalize
      final truncatedCandidates = candidates.sublist(0, cutoff);
      final truncatedExps = exps.sublist(0, cutoff);
      var truncatedSum = 0.0;
      for (final e in truncatedExps) {
        truncatedSum += e;
      }
      for (var i = 0; i < truncatedExps.length; i++) {
        truncatedExps[i] /= truncatedSum;
      }
      return _sampleFromDistribution(truncatedCandidates, truncatedExps);
    }

    return _sampleFromDistribution(candidates, exps);
  }

  /// Returns the index of the maximum value in [values].
  int _argmax(Float32List values) {
    var maxIdx = 0;
    var maxVal = values[0];
    for (var i = 1; i < values.length; i++) {
      if (values[i] > maxVal) {
        maxVal = values[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  /// Samples a token from a weighted distribution.
  int _sampleFromDistribution(List<int> tokenIds, List<double> probs) {
    final r = _random.nextDouble();
    var cumulative = 0.0;
    for (var i = 0; i < tokenIds.length; i++) {
      cumulative += probs[i];
      if (r < cumulative) return tokenIds[i];
    }
    return tokenIds.last;
  }
}
