/// Configuration for controlling text generation behavior.
///
/// Provides knobs for temperature, sampling strategy, and stopping criteria.
class GenerationConfig {
  /// Maximum number of new tokens to generate.
  final int maxTokens;

  /// Temperature for logit scaling. Higher values = more random.
  ///
  /// A value of `0.0` is equivalent to greedy (argmax) decoding.
  /// Typical values range from `0.1` to `1.5`.
  final double temperature;

  /// Top-P (nucleus) sampling threshold.
  ///
  /// Only the smallest set of tokens whose cumulative probability exceeds [topP]
  /// are considered for sampling. Set to `1.0` to disable.
  final double topP;

  /// Top-K sampling threshold.
  ///
  /// Only the [topK] tokens with the highest probabilities are considered.
  /// Set to `0` to disable.
  final int topK;

  /// Repetition penalty factor.
  ///
  /// Values greater than `1.0` penalize tokens that have already appeared.
  /// Set to `1.0` to disable.
  final double repetitionPenalty;

  /// Token IDs that signal the end of generation.
  ///
  /// If null, the model's default EOS token from [ModelConfig] is used.
  final List<int>? stopTokenIds;

  const GenerationConfig({
    this.maxTokens = 256,
    this.temperature = 1.0,
    this.topP = 1.0,
    this.topK = 0,
    this.repetitionPenalty = 1.0,
    this.stopTokenIds,
  });

  /// Greedy decoding (always picks the most likely token).
  static const greedy = GenerationConfig(temperature: 0.0);
}
