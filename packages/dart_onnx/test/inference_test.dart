import 'dart:typed_data';
import 'package:dart_onnx/dart_onnx.dart';
import 'package:dart_onnx/src/ffi/ort_bindings.dart';
import 'package:test/test.dart';

void main() {
  group('DartONNXSession Inference', () {
    late DartONNX env;

    setUpAll(() {
      env = DartONNX();
    });

    tearDownAll(() {
      env.dispose();
    });

    test('Runs Identity Model correctly', () {
      final session = DartONNXSession.fromFile(
        env,
        'test/assets/models/identity.onnx',
      );

      // shape: [1, 2, 2]
      final inputTensor = DartONNXTensor.float32(
        data: Float32List.fromList([1.0, 2.5, 3.3, 4.7]),
        shape: [1, 2, 2],
      );

      final results = session.run({'X': inputTensor});
      expect(results.length, 1);
      final outputTensor = results['Y']!;

      // Verify metadata extracted via fromOrtValue
      expect(outputTensor.shape, [1, 2, 2]);
      expect(
        outputTensor.elementType,
        ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      );

      final outData = outputTensor.data as Float32List;

      expect(outData[0], closeTo(1.0, 0.0001));
      expect(outData[1], closeTo(2.5, 0.0001));
      expect(outData[2], closeTo(3.3, 0.0001));
      expect(outData[3], closeTo(4.7, 0.0001));

      inputTensor.dispose();
      session.dispose();
    });

    test('Runs Add Model correctly', () {
      final session = DartONNXSession.fromFile(
        env,
        'test/assets/models/add.onnx',
      );

      // shape: [3]
      final tensorA = DartONNXTensor.float32(
        data: Float32List.fromList([10.0, 20.0, 30.0]),
        shape: [3],
      );

      final tensorB = DartONNXTensor.float32(
        data: Float32List.fromList([5.0, 3.0, 1.0]),
        shape: [3],
      );

      final results = session.run({'A': tensorA, 'B': tensorB});
      expect(results.length, 1);

      final outputTensor = results['C']!;
      final outData = outputTensor.data as Float32List;

      expect(outData[0], closeTo(15.0, 0.0001));
      expect(outData[1], closeTo(23.0, 0.0001));
      expect(outData[2], closeTo(31.0, 0.0001));

      tensorA.dispose();
      tensorB.dispose();
      session.dispose();
    });

    test('Runs Identity Model with Float inputs', () {
      final session = DartONNXSession.fromFile(
        env,
        'test/assets/models/identity.onnx',
      );

      final inputTensor = DartONNXTensor.float32(
        data: Float32List.fromList([100.0, -200.0, 300.0, 400.0]),
        shape: [1, 2, 2],
      );

      final results = session.run({'X': inputTensor});
      final outputTensor = results['Y']!;

      expect(
        outputTensor.elementType,
        ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      );

      final outData = outputTensor.data as Float32List;
      expect(outData, [100.0, -200.0, 300.0, 400.0]);

      inputTensor.dispose();
      session.dispose();
    });

    test('Runs Multi-Output Model correctly', () {
      final session = DartONNXSession.fromFile(
        env,
        'test/assets/models/multi_output.onnx',
      );

      final inputTensor = DartONNXTensor.float32(
        data: Float32List.fromList([42.0, 84.0]),
        shape: [1, 2],
      );

      final results = session.run({'X': inputTensor});
      expect(results.length, 2);
      expect(results.containsKey('Y'), isTrue);
      expect(results.containsKey('Z'), isTrue);

      final outY = results['Y']!;
      final outZ = results['Z']!;

      expect(outY.shape, [1, 2]);
      expect(outZ.shape, [1, 2]);

      expect((outY.data as Float32List), [42.0, 84.0]);
      expect((outZ.data as Float32List), [42.0, 84.0]);

      inputTensor.dispose();
      session.dispose();
    });
  });
}
