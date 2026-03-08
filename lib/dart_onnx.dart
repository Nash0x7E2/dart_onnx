/// DartONNX — Cross-platform ONNX model inference for Dart.
///
/// A high-performance Dart package for running ONNX models using
/// ONNX Runtime via Dart FFI.
///
/// ## Getting Started
///
/// ```dart
/// import 'package:dart_onnx/dart_onnx.dart';
///
/// void main() {
///   final env = DartONNX(loggingLevel: DartONNXLoggingLevel.warning);
///   final session = DartONNXSession.fromFile(env, 'model.onnx');
///   final outputs = session.run({'input': myTensor});
///   print(outputs);
/// }
/// ```
library dart_onnx;

export 'src/dart_onnx_env.dart';
export 'src/dart_onnx_session.dart';
export 'src/dart_onnx_tensor.dart';
export 'src/dart_onnx_execution_provider.dart';
export 'src/dart_onnx_logging_level.dart';
export 'src/dart_onnx_exception.dart';
