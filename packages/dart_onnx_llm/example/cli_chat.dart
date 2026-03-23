/// Example: Interactive CLI chat using dart_onnx_llm.
///
/// This example loads the SmolLM2-135M ONNX model and runs an interactive
/// chat session in the terminal.
///
/// ## Setup
///
/// First, download the model files:
///
/// ```
/// dart run tool/download_model.dart
/// ```
///
/// Then run this example from the package root:
///
/// ```
/// dart run example/cli_chat.dart
/// ```
library;

import 'dart:io';

import 'package:dart_onnx/dart_onnx.dart';
import 'package:dart_onnx_llm/dart_onnx_llm.dart';

void main() async {
  final scriptDir = File.fromUri(Platform.script).parent;
  final modelDir = '${scriptDir.path}/model';

  // Check that model files exist.
  if (!File('$modelDir/config.json').existsSync() ||
      !File('$modelDir/tokenizer.json').existsSync()) {
    stderr.writeln('Model files not found at: $modelDir');
    stderr.writeln(
      'Run `dart run tool/download_model.dart` first to download them.',
    );
    exit(1);
  }

  print('╔════════════════════════════════════════════════╗');
  print('║         dart_onnx_llm — CLI Chat Demo         ║');
  print('║  Loading SmolLM2-135M... (this may take a moment)  ║');
  print('╚════════════════════════════════════════════════╝');
  print('');

  // Load the pipeline from the model directory.
  final pipeline = await TextGenerationPipeline.fromDirectory(
    modelDir,
    executionProviders: [
      DartONNXExecutionProvider.coreML, // Apple Neural Engine (if available)
      DartONNXExecutionProvider.cpu, // Fallback
    ],
  );

  print('✓ Model loaded successfully!');
  print('');
  print('Type your messages below. Type "exit" or "quit" to stop.');
  print('Type "clear" to reset the conversation.');
  print('─' * 50);
  print('');

  // Create a stateful chat session.
  final session = pipeline.createChatSession(
    systemPrompt: 'You are a helpful, friendly, and concise AI assistant.',
    config: GenerationConfig(
      maxTokens: 128,
      temperature: 0.7,
      topP: 0.9,
    ),
  );

  // Interactive loop.
  while (true) {
    stdout.write('You: ');
    final input = stdin.readLineSync();

    if (input == null || input.trim().isEmpty) continue;

    final trimmed = input.trim();
    if (trimmed.toLowerCase() == 'exit' || trimmed.toLowerCase() == 'quit') {
      print('\nGoodbye! 👋');
      break;
    }

    if (trimmed.toLowerCase() == 'clear') {
      session.clearHistory();
      print('[Conversation cleared]\n');
      continue;
    }

    stdout.write('Assistant: ');

    final stopwatch = Stopwatch()..start();
    var tokenCount = 0;

    await for (final chunk in session.sendMessageStream(trimmed)) {
      stdout.write(chunk);
      tokenCount++;
    }

    stopwatch.stop();

    print('');
    print(
      '  [$tokenCount tokens in ${stopwatch.elapsedMilliseconds}ms '
      '| ${(tokenCount / (stopwatch.elapsedMilliseconds / 1000)).toStringAsFixed(1)} tok/s]',
    );
    print('');
  }

  // Cleanup.
  pipeline.dispose();
}
