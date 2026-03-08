import 'dart:typed_data';
import 'package:dart_onnx/dart_onnx.dart';

/// Example demonstrating how to use the DartONNX package
/// to run inference on an ONNX model.
void main() {
  // 1. Initialize the Runtime Environment
  final env = DartONNX(loggingLevel: DartONNXLoggingLevel.warning);
  print('ONNX Runtime version: ${env.ortVersion}');

  // 2. Load the model specifying the execution providers via Enums.
  final session = DartONNXSession.fromFile(
    env,
    'model.onnx',
    executionProviders: [
      DartONNXExecutionProvider.coreML, // Apple Neural Engine
      DartONNXExecutionProvider.cpu, // Fallback
    ],
  );

  print('Session inputs: ${session.inputNames}');
  print('Session outputs: ${session.outputNames}');

  // 3. Prepare Input Tensor
  final inputData = Float32List.fromList([1.0, 2.0, 3.0]);
  final tensor = DartONNXTensor.float32(data: inputData, shape: [1, 3]);

  // 4. Run the inference
  final outputs = session.run({'input': tensor});

  // 5. Read the output
  final outputTensor = outputs.values.first;
  print('Output: ${outputTensor.data}');
  print('Output shape: ${outputTensor.shape}');
}
