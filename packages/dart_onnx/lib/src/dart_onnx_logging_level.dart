/// The logging level for the ONNX Runtime environment.
enum DartONNXLoggingLevel {
  /// Verbose logging.
  verbose(0),

  /// Informational logging.
  info(1),

  /// Warning logging.
  warning(2),

  /// Error logging.
  error(3),

  /// Fatal logging.
  fatal(4);

  /// The integer value expected by ORT C API.
  final int value;

  const DartONNXLoggingLevel(this.value);
}
