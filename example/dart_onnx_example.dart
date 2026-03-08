/// Example: Running SmolLM2-135M inference with dart_onnx.
///
/// This example loads the quantized SmolLM2-135M ONNX model and runs a single
/// forward pass to predict the next token for a short prompt.
///
/// ## Setup
///
/// First, download the model files (≈138 MB total):
///
/// ```
/// dart run tool/download_model.dart
/// ```
///
/// Then run this example from the package root:
///
/// ```
/// dart run example/dart_onnx_example.dart
/// ```
library;

import 'dart:io';
import 'dart:typed_data';
import 'package:dart_onnx/dart_onnx.dart';

// ---------------------------------------------------------------------------
// Model configuration (matches config.json from HuggingFace)
// ---------------------------------------------------------------------------

/// Number of transformer layers (num_hidden_layers).
const int kNumLayers = 30;

/// Number of key/value attention heads (num_key_value_heads).
const int kNumKvHeads = 3;

/// Dimension per attention head (head_dim).
const int kHeadDim = 64;

/// Vocabulary size.
const int kVocabSize = 49152;

/// End-of-sequence token ID.
const int kEosTokenId = 0;

// ---------------------------------------------------------------------------
// A minimal hard-coded tokenization of the prompt "Hello, I am".
//
// These IDs were looked up in SmolLM2's GPT-2-style BPE vocabulary
// (tokenizer.json) and are stable for this exact prompt.
//
// In a real application you would use a proper tokenizer library that reads
// `tokenizer.json` to tokenize arbitrary text.
// ---------------------------------------------------------------------------
const List<int> kPromptTokenIds = [
  12906, // "Hello"
  13, // ","
  309, // " I"
  837, // " am"
];

void main() {
  // ── 1. Locate model file ──────────────────────────────────────────────────
  final scriptDir = File.fromUri(Platform.script).parent;
  final modelPath = '${scriptDir.path}/onnx/model_quantized.onnx';

  if (!File(modelPath).existsSync()) {
    stderr.writeln('Model not found at: $modelPath');
    stderr.writeln(
      'Run `dart run tool/download_model.dart` first to download it.',
    );
    exit(1);
  }

  // ── 2. Initialize ONNX Runtime environment ────────────────────────────────
  print('Initializing ONNX Runtime...');
  final env = DartONNX(loggingLevel: DartONNXLoggingLevel.warning);
  print('ORT version : ${env.ortVersion}');

  // ── 3. Load the session ───────────────────────────────────────────────────
  print('Loading model: $modelPath');
  final session = DartONNXSession.fromFile(
    env,
    modelPath,
    executionProviders: [
      DartONNXExecutionProvider.coreML, // Apple Neural Engine (optional)
      DartONNXExecutionProvider.cpu, // Fallback
    ],
  );

  print('Input  names : ${session.inputNames}');
  print('Output names : ${session.outputNames}');

  // ── 4. Build input tensors ────────────────────────────────────────────────
  final seqLen = kPromptTokenIds.length;
  const int batch = 1;

  // input_ids  [1, seqLen]  — the token IDs of the prompt
  final inputIds = DartONNXTensor.int64(
    data: Int64List.fromList(kPromptTokenIds),
    shape: [batch, seqLen],
  );

  // attention_mask  [1, seqLen]  — all 1s (no padding)
  final attentionMask = DartONNXTensor.int64(
    data: Int64List.fromList(List.filled(seqLen, 1)),
    shape: [batch, seqLen],
  );

  // position_ids  [1, seqLen]  — 0, 1, 2, …, seqLen-1
  final positionIds = DartONNXTensor.int64(
    data: Int64List.fromList(List.generate(seqLen, (i) => i)),
    shape: [batch, seqLen],
  );

  // Build the input map with only the tensors the model expects.
  // If the session also expects past_key_values we provide zero tensors.
  final inputs = <String, DartONNXTensor>{
    'input_ids': inputIds,
    'attention_mask': attentionMask,
  };

  if (session.inputNames.contains('position_ids')) {
    inputs['position_ids'] = positionIds;
  }

  // Provide empty (zero) past_key_values for layer 0..N-1 when needed.
  // Shape: [batch, num_kv_heads, 0, head_dim]  — sequence dim = 0 means "empty cache".
  final kvInputNames = session.inputNames
      .where((n) => n.startsWith('past_key_values.'))
      .toList();

  if (kvInputNames.isNotEmpty) {
    print('Providing ${kvInputNames.length} empty past_key_value tensors...');
    for (final name in kvInputNames) {
      inputs[name] = DartONNXTensor.float32(
        data: Float32List(0),
        shape: [batch, kNumKvHeads, 0, kHeadDim],
      );
    }
  }

  // ── 5. Run inference ──────────────────────────────────────────────────────
  print('\nRunning forward pass on prompt: "Hello, I am"');
  print('Prompt token IDs: $kPromptTokenIds\n');

  final stopwatch = Stopwatch()..start();
  final outputs = session.run(inputs);
  stopwatch.stop();

  print('Inference time: ${stopwatch.elapsedMilliseconds} ms');

  // ── 6. Read logits and show top-5 next-token predictions ─────────────────
  final logitsTensor = outputs['logits'];
  if (logitsTensor == null) {
    throw StateError('Expected "logits" in outputs but got: ${outputs.keys}');
  }

  // logits shape: [batch=1, seq_len, vocab_size]
  // We want the logits at the last token position → index [0, seqLen-1, :].
  final logitsData = logitsTensor.data as Float32List;
  final logitsShape = logitsTensor.shape;

  print('Logits shape  : $logitsShape');

  // Slice out last-token logits.
  final lastTokenOffset = (seqLen - 1) * kVocabSize;
  final lastTokenLogits = logitsData.sublist(
    lastTokenOffset,
    lastTokenOffset + kVocabSize,
  );

  // Compute top-5 by brute-force sort (vocab is small enough).
  final indexed = List.generate(kVocabSize, (i) => (i, lastTokenLogits[i]))
    ..sort((a, b) => b.$2.compareTo(a.$2));

  print('\nTop-5 next-token predictions (token_id → logit_score):');
  for (final (tokenId, score) in indexed.take(5)) {
    print('  token $tokenId → ${score.toStringAsFixed(4)}');
  }

  // ── 7. Greedy next-token ──────────────────────────────────────────────────
  final nextTokenId = indexed.first.$1;
  print('\nGreedy next token ID : $nextTokenId');
  if (nextTokenId == kEosTokenId) {
    print('(EOS token — model wants to stop here)');
  }
  print('Decode with `tokenizer.json` to convert token IDs back to a string.');

  // ── 8. Cleanup ────────────────────────────────────────────────────────────
  for (final t in inputs.values) {
    t.dispose();
  }
  for (final t in outputs.values) {
    t.dispose();
  }
  session.dispose();

  print('\nDone.');
}
