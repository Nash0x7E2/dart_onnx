/// The hardware execution provider to use for running ONNX models.
///
/// Pass one or more of these to [DartONNXSession.fromFile] or
/// [DartONNXSession.fromBytes]. ONNX Runtime will attempt to use
/// them in the order they are provided, falling back to the next
/// one if the current one is unavailable.
enum DartONNXExecutionProvider {
  /// CPU execution (always available).
  cpu('CPUExecutionProvider'),

  /// Apple CoreML (iOS / macOS).
  coreML('CoreMLExecutionProvider'),

  /// Android NNAPI.
  nnapi('NnapiExecutionProvider'),

  /// NVIDIA CUDA.
  cuda('CUDAExecutionProvider'),

  /// NVIDIA TensorRT.
  tensorRT('TensorrtExecutionProvider'),

  /// AMD ROCm.
  rocm('ROCMExecutionProvider'),

  /// Intel OpenVINO.
  openVINO('OpenVINOExecutionProvider'),

  /// DirectML (Windows).
  directML('DmlExecutionProvider'),

  /// Qualcomm QNN.
  qnn('QNNExecutionProvider'),

  /// XNNPACK.
  xnnpack('XnnpackExecutionProvider');

  /// The provider name as expected by the ORT C API.
  final String ortName;

  const DartONNXExecutionProvider(this.ortName);
}
