import 'package:dart_onnx/dart_onnx.dart';
import 'package:test/test.dart';

void main() {
  group('DartONNX', () {
    test('DartONNXExecutionProvider has correct ORT names', () {
      expect(DartONNXExecutionProvider.cpu.ortName, 'CPUExecutionProvider');
      expect(
        DartONNXExecutionProvider.coreML.ortName,
        'CoreMLExecutionProvider',
      );
      expect(DartONNXExecutionProvider.cuda.ortName, 'CUDAExecutionProvider');
    });

    test('DartONNXLoggingLevel has correct values', () {
      expect(DartONNXLoggingLevel.verbose.value, 0);
      expect(DartONNXLoggingLevel.warning.value, 2);
      expect(DartONNXLoggingLevel.fatal.value, 4);
    });
  });
}
