import 'dart:typed_data';
import 'package:dart_onnx/dart_onnx.dart';
import 'package:test/test.dart';

void main() {
  group('DartONNX Stress Tests', () {
    late DartONNX env;

    setUpAll(() {
      env = DartONNX();
    });

    tearDownAll(() {
      env.dispose();
    });

    test('Rapid tensor creation and disposal', () {
      for (var i = 0; i < 10000; i++) {
        final tensor = DartONNXTensor.float32(
          data: Float32List.fromList(List.filled(100, 0.0)),
          shape: [10, 10],
        );
        // Disposing shouldn't crash
        tensor.dispose();
      }
    });

    test('Rapid session loading and inference', () {
      final session = DartONNXSession.fromFile(
        env,
        'test/assets/models/add.onnx',
      );

      for (var i = 0; i < 1000; i++) {
        final tensorA = DartONNXTensor.float32(
          data: Float32List.fromList([
            i.toDouble(),
            i.toDouble(),
            i.toDouble(),
          ]),
          shape: [3],
        );
        final tensorB = DartONNXTensor.float32(
          data: Float32List.fromList([0.0, 1.0, 2.0]),
          shape: [3],
        );

        final res = session.run({'A': tensorA, 'B': tensorB});
        final outTensor = res['C']!;
        expect(
          (outTensor.data as Float32List)[0],
          closeTo(i.toDouble(), 0.0001),
        );

        tensorA.dispose();
        tensorB.dispose();
        outTensor.dispose();
      }

      session.dispose();
    });
  });
}
