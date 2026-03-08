/// Exception thrown by the DartONNX package when an ONNX Runtime operation fails.
class DartONNXException implements Exception {
  /// The error message from ONNX Runtime.
  final String message;

  /// The ORT error code, if available.
  final int? errorCode;

  DartONNXException(this.message, {this.errorCode});

  @override
  String toString() =>
      'DartONNXException: $message${errorCode != null ? ' (code: $errorCode)' : ''}';
}
