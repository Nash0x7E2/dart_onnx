import 'dart:typed_data';
import 'package:dart_onnx/dart_onnx.dart';
import 'package:test/test.dart';

void main() {
  group('DartONNXTensor Types Memory', () {
    test('Allocates Float32 correctly', () {
      final tensorFloat = DartONNXTensor.float32(
        data: Float32List.fromList([0, 0, 0, 0]),
        shape: [2, 2],
      );
      expect(tensorFloat.shape, [2, 2]);
      expect(tensorFloat.data, isA<Float32List>());
      expect((tensorFloat.data as Float32List), [0.0, 0.0, 0.0, 0.0]);

      // Can update data inside
      final list = tensorFloat.data as Float32List;
      list[1] = 5.0;
      expect((tensorFloat.data as Float32List)[1], 5.0);
      tensorFloat.dispose();
    });

    test('Allocates Int64 correctly', () {
      final tensorInt = DartONNXTensor.int64(
        data: Int64List.fromList([0, 0, 0]),
        shape: [3],
      );
      expect(tensorInt.shape, [3]);
      expect((tensorInt.data as Int64List), [0, 0, 0]);

      final list = tensorInt.data as Int64List;
      list[2] = 42;
      expect((tensorInt.data as Int64List)[2], 42);
      tensorInt.dispose();
    });
  });

  group('DartONNXTensor — All Types', () {
    test('Float64 round-trip', () {
      final tensor = DartONNXTensor.float64(
        data: Float64List.fromList([1.5, -2.25]),
        shape: [2],
      );
      expect(tensor.shape, [2]);
      final data = tensor.data;
      expect(data, isA<Float64List>());
      expect(data as Float64List, [1.5, -2.25]);
      tensor.dispose();
    });

    test('Int32 round-trip', () {
      final tensor = DartONNXTensor.int32(
        data: Int32List.fromList([42, -99]),
        shape: [1, 2],
      );
      expect(tensor.shape, [1, 2]);
      final data = tensor.data;
      expect(data, isA<Int32List>());
      expect(data as Int32List, [42, -99]);
      tensor.dispose();
    });

    test('Uint8 round-trip', () {
      final tensor = DartONNXTensor.uint8(
        data: Uint8List.fromList([0, 128, 255]),
        shape: [3],
      );
      expect(tensor.shape, [3]);
      final data = tensor.data;
      expect(data, isA<Uint8List>());
      expect(data as Uint8List, [0, 128, 255]);
      tensor.dispose();
    });
  });

  group('DartONNXTensor Arena Management', () {
    test('dispose() frees independent memory', () {
      final tensor = DartONNXTensor.float32(
        data: Float32List.fromList([0, 0, 0]),
        shape: [3],
      );
      // Explicitly dispose and verify we don't crash
      tensor.dispose();
    });

    test('dispose() is idempotent', () {
      final tensor = DartONNXTensor.float32(
        data: Float32List.fromList([1]),
        shape: [1],
      );
      tensor.dispose();
      tensor.dispose(); // Should not throw
    });
  });

  group('DartONNXTensor — Use-After-Dispose', () {
    test('Calling pointer after dispose throws StateError', () {
      final tensor = DartONNXTensor.float32(
        data: Float32List.fromList([1]),
        shape: [1],
      );
      tensor.dispose();
      expect(() => tensor.pointer, throwsStateError);
    });

    test('Calling data after dispose throws StateError', () {
      final tensor = DartONNXTensor.float32(
        data: Float32List.fromList([1]),
        shape: [1],
      );
      tensor.dispose();
      // internally data calls pointer, so it throws StateError
      expect(() => tensor.data, throwsStateError);
    });
  });
}
