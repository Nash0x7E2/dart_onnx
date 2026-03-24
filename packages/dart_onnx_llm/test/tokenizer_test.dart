import 'dart:io';

import 'package:dart_onnx_llm/dart_onnx_llm.dart';
import 'package:test/test.dart';

/// These expected token IDs were generated using the Python transformers lib:
///
/// ```python
/// from transformers import AutoTokenizer
/// t = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M')
/// t.encode(text, add_special_tokens=False)
/// ```
///
/// If these tests pass, our Dart tokenizer produces identical output to
/// the HuggingFace reference implementation.
void main() {
  late Tokenizer tokenizer;

  setUpAll(() async {
    // Try to load the tokenizer.json from the model directory.
    // The download_model.dart tool handles downloading this file.
    final modelDir = _findModelDir();
    if (modelDir == null) {
      fail(
        'Could not find tokenizer.json. '
        'Run `dart run tool/download_model.dart` first.',
      );
    }
    tokenizer = await Tokenizer.fromFile('$modelDir/tokenizer.json');
  });

  group('Tokenizer encode matches Python reference', () {
    test('simple greeting', () {
      // 'Hello, I am' => [19556, 28, 339, 744]
      expect(tokenizer.encode('Hello, I am'), equals([19556, 28, 339, 744]));
    });

    test('punctuation', () {
      // 'Hello, world!' => [19556, 28, 905, 17]
      expect(tokenizer.encode('Hello, world!'), equals([19556, 28, 905, 17]));
    });

    test('full sentence', () {
      // 'The capital of France is Paris.' => [504, 3575, 282, 4649, 314, 7042, 30]
      expect(
        tokenizer.encode('The capital of France is Paris.'),
        equals([504, 3575, 282, 4649, 314, 7042, 30]),
      );
    });

    test('digits are split individually', () {
      // 'I have 42 cats and 3 dogs.' =>
      //   [57, 457, 216, 36, 34, 7680, 284, 216, 35, 5046, 30]
      expect(
        tokenizer.encode('I have 42 cats and 3 dogs.'),
        equals([57, 457, 216, 36, 34, 7680, 284, 216, 35, 5046, 30]),
      );
    });

    test('multiple spaces', () {
      // '  Multiple   spaces  here' => [216, 16560, 256, 5600, 216, 1535]
      expect(
        tokenizer.encode('  Multiple   spaces  here'),
        equals([216, 16560, 256, 5600, 216, 1535]),
      );
    });

    test('contractions', () {
      // "can't won't don't" => [4055, 982, 3763, 982, 1326, 982]
      expect(
        tokenizer.encode("can't won't don't"),
        equals([4055, 982, 3763, 982, 1326, 982]),
      );
    });

    test('code-like text', () {
      // 'def hello_world():' => [1604, 33662, 79, 6693, 3734]
      expect(
        tokenizer.encode('def hello_world():'),
        equals([1604, 33662, 79, 6693, 3734]),
      );
    });

    test('numbers with operators', () {
      // 'x = 123 + 456' =>
      //   [104, 446, 216, 33, 34, 35, 1232, 216, 36, 37, 38]
      expect(
        tokenizer.encode('x = 123 + 456'),
        equals([104, 446, 216, 33, 34, 35, 1232, 216, 36, 37, 38]),
      );
    });

    test('special tokens are encoded correctly', () {
      // '<|im_start|>system' => [1, 9690]
      expect(tokenizer.encode('<|im_start|>system'), equals([1, 9690]));
    });
  });

  group('Tokenizer decode', () {
    test('round-trip encode/decode', () {
      const text = 'Hello, world!';
      final ids = tokenizer.encode(text);
      expect(tokenizer.decode(ids), equals(text));
    });

    test('round-trip with numbers', () {
      const text = 'I have 42 cats.';
      final ids = tokenizer.encode(text);
      expect(tokenizer.decode(ids), equals(text));
    });

    test('round-trip with contractions', () {
      const text = "can't won't don't";
      final ids = tokenizer.encode(text);
      expect(tokenizer.decode(ids), equals(text));
    });
  });

  group('Tokenizer properties', () {
    test('vocab size is reasonable', () {
      // SmolLM2-135M has a ~49k vocab
      expect(tokenizer.vocabSize, greaterThan(40000));
      expect(tokenizer.vocabSize, lessThan(60000));
    });

    test('empty string encodes to empty list', () {
      expect(tokenizer.encode(''), isEmpty);
    });

    test('idToToken and tokenToId are consistent', () {
      // Spot check a few known tokens
      final helloId = tokenizer.tokenToId('Hello');
      expect(helloId, isNotNull);
      expect(tokenizer.idToToken(helloId!), equals('Hello'));
    });
  });
}

/// Tries to find the model directory containing tokenizer.json.
String? _findModelDir() {
  // Check relative to the test file location (standard package layout).
  final candidates = ['example/model', '../example/model', 'test/fixtures'];

  for (final candidate in candidates) {
    if (File('$candidate/tokenizer.json').existsSync()) {
      return candidate;
    }
  }

  return null;
}
