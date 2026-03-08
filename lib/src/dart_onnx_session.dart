import 'dart:ffi';
import 'package:ffi/ffi.dart';

import 'ffi/ort_bindings.dart';
import 'ort_ffi.dart';
import 'dart_onnx_env.dart';
import 'dart_onnx_execution_provider.dart';
import 'dart_onnx_tensor.dart';

/// An ONNX inference session.
///
/// Loads an ONNX model and provides methods to inspect its inputs/outputs
/// and run inferences.
///
/// ```dart
/// final session = DartONNXSession.fromFile(
///   env,
///   'model.onnx',
///   executionProviders: [DartONNXExecutionProvider.cpu],
/// );
///
/// final outputs = session.run({'input': inputTensor});
/// ```
class DartONNXSession implements Finalizable {
  static final _finalizer = NativeFinalizer(_releaseFn);

  static final Pointer<NativeFunction<Void Function(Pointer<Void>)>>
  _releaseFn = OrtFFI.instance.api.ref.ReleaseSession.cast();

  /// Raw pointer to the OrtSession.
  final Pointer<OrtSession> _ptr;

  /// Cached input names.
  late final List<String> inputNames;

  /// Cached output names.
  late final List<String> outputNames;

  bool _disposed = false;

  DartONNXSession._(this._ptr) {
    _finalizer.attach(this, _ptr.cast(), detach: this);
    inputNames = _getInputNames();
    outputNames = _getOutputNames();
  }

  /// Get the raw OrtSession pointer. Throws if already disposed.
  Pointer<OrtSession> get pointer {
    if (_disposed) {
      throw StateError('DartONNXSession has been disposed.');
    }
    return _ptr;
  }

  /// Create a session from a model file on disk.
  ///
  /// [env] is the [DartONNX] environment.
  /// [modelPath] is the path to the `.onnx` model file.
  /// [executionProviders] specifies the hardware accelerators to use (in priority order).
  factory DartONNXSession.fromFile(
    DartONNX env,
    String modelPath, {
    List<DartONNXExecutionProvider> executionProviders = const [
      DartONNXExecutionProvider.cpu,
    ],
  }) {
    final ort = OrtFFI.instance;
    final api = ort.api.ref;

    // Create session options
    final sessionOptions = _createSessionOptions(ort, executionProviders);

    // Create the session
    final createSession =
        api.CreateSession.asFunction<
          Pointer<OrtStatus> Function(
            Pointer<OrtEnv>,
            Pointer<Char>,
            Pointer<OrtSessionOptions>,
            Pointer<Pointer<OrtSession>>,
          )
        >();

    final pathNative = modelPath.toNativeUtf8().cast<Char>();
    final outPtr = calloc<Pointer<OrtSession>>();

    try {
      final status = createSession(
        env.pointer,
        pathNative,
        sessionOptions,
        outPtr,
      );
      ort.checkStatus(status);
      return DartONNXSession._(outPtr.value);
    } finally {
      calloc.free(pathNative);
      calloc.free(outPtr);
      final releaseOpts =
          api.ReleaseSessionOptions.asFunction<
            void Function(Pointer<OrtSessionOptions>)
          >();
      releaseOpts(sessionOptions);
    }
  }

  /// Create a session from a model in memory (as bytes).
  ///
  /// [env] is the [DartONNX] environment.
  /// [modelBytes] is the raw ONNX model data.
  /// [executionProviders] specifies the hardware accelerators to use (in priority order).
  factory DartONNXSession.fromBytes(
    DartONNX env,
    List<int> modelBytes, {
    List<DartONNXExecutionProvider> executionProviders = const [
      DartONNXExecutionProvider.cpu,
    ],
  }) {
    final ort = OrtFFI.instance;
    final api = ort.api.ref;

    // Create session options
    final sessionOptions = _createSessionOptions(ort, executionProviders);

    // Copy model bytes to native memory
    final modelData = calloc<Uint8>(modelBytes.length);
    for (var i = 0; i < modelBytes.length; i++) {
      modelData[i] = modelBytes[i];
    }

    // Create the session
    final createSession =
        api.CreateSessionFromArray.asFunction<
          Pointer<OrtStatus> Function(
            Pointer<OrtEnv>,
            Pointer<Void>,
            int,
            Pointer<OrtSessionOptions>,
            Pointer<Pointer<OrtSession>>,
          )
        >();

    final outPtr = calloc<Pointer<OrtSession>>();

    try {
      final status = createSession(
        env.pointer,
        modelData.cast(),
        modelBytes.length,
        sessionOptions,
        outPtr,
      );
      ort.checkStatus(status);
      return DartONNXSession._(outPtr.value);
    } finally {
      calloc.free(modelData);
      calloc.free(outPtr);
      final releaseOpts =
          api.ReleaseSessionOptions.asFunction<
            void Function(Pointer<OrtSessionOptions>)
          >();
      releaseOpts(sessionOptions);
    }
  }

  /// Run inference on the model.
  ///
  /// [inputs] maps input names to their [DartONNXTensor] values.
  /// Returns a map of output names to their result [DartONNXTensor] values.
  Map<String, DartONNXTensor> run(Map<String, DartONNXTensor> inputs) {
    final ort = OrtFFI.instance;
    final api = ort.api.ref;

    // Validate inputs
    for (final name in inputs.keys) {
      if (!inputNames.contains(name)) {
        throw ArgumentError(
          'Unknown input name "$name". Expected one of: $inputNames',
        );
      }
    }

    final inputCount = inputs.length;
    final outputCount = outputNames.length;

    // Prepare input names
    final inputNamesPtr = calloc<Pointer<Char>>(inputCount);
    final inputNamesNative = <Pointer<Utf8>>[];
    var i = 0;
    final orderedInputNames = <String>[];
    for (final entry in inputs.entries) {
      final namePtr = entry.key.toNativeUtf8();
      inputNamesNative.add(namePtr);
      inputNamesPtr[i] = namePtr.cast();
      orderedInputNames.add(entry.key);
      i++;
    }

    // Prepare input values
    final inputValuesPtr = calloc<Pointer<OrtValue>>(inputCount);
    i = 0;
    for (final entry in inputs.entries) {
      inputValuesPtr[i] = entry.value.pointer;
      i++;
    }

    // Prepare output names
    final outputNamesPtr = calloc<Pointer<Char>>(outputCount);
    final outputNamesNative = <Pointer<Utf8>>[];
    for (var j = 0; j < outputCount; j++) {
      final namePtr = outputNames[j].toNativeUtf8();
      outputNamesNative.add(namePtr);
      outputNamesPtr[j] = namePtr.cast();
    }

    // Prepare output values (will be filled by ORT)
    final outputValuesPtr = calloc<Pointer<OrtValue>>(outputCount);
    for (var j = 0; j < outputCount; j++) {
      outputValuesPtr[j] = nullptr;
    }

    // Run inference
    final runFn =
        api.Run.asFunction<
          Pointer<OrtStatus> Function(
            Pointer<OrtSession>,
            Pointer<OrtRunOptions>,
            Pointer<Pointer<Char>>,
            Pointer<Pointer<OrtValue>>,
            int,
            Pointer<Pointer<Char>>,
            int,
            Pointer<Pointer<OrtValue>>,
          )
        >();

    try {
      final status = runFn(
        pointer,
        nullptr, // default run options
        inputNamesPtr,
        inputValuesPtr,
        inputCount,
        outputNamesPtr,
        outputCount,
        outputValuesPtr,
      );
      ort.checkStatus(status);

      // Build output map
      final results = <String, DartONNXTensor>{};
      for (var j = 0; j < outputCount; j++) {
        results[outputNames[j]] = DartONNXTensor.fromOrtValue(
          outputValuesPtr[j],
        );
      }
      return results;
    } finally {
      // Free native strings
      for (final ptr in inputNamesNative) {
        calloc.free(ptr);
      }
      for (final ptr in outputNamesNative) {
        calloc.free(ptr);
      }
      calloc.free(inputNamesPtr);
      calloc.free(inputValuesPtr);
      calloc.free(outputNamesPtr);
      calloc.free(outputValuesPtr);
    }
  }

  /// Manually dispose the session's native resources.
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _finalizer.detach(this);
    final release = OrtFFI.instance.api.ref.ReleaseSession
        .asFunction<void Function(Pointer<OrtSession>)>();
    release(_ptr);
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Private helpers
  // ──────────────────────────────────────────────────────────────────────────

  /// Create an OrtSessionOptions with the given execution providers.
  static Pointer<OrtSessionOptions> _createSessionOptions(
    OrtFFI ort,
    List<DartONNXExecutionProvider> providers,
  ) {
    final api = ort.api.ref;

    final createOpts =
        api.CreateSessionOptions.asFunction<
          Pointer<OrtStatus> Function(Pointer<Pointer<OrtSessionOptions>>)
        >();
    final optsPtr = calloc<Pointer<OrtSessionOptions>>();
    ort.checkStatus(createOpts(optsPtr));
    final opts = optsPtr.value;
    calloc.free(optsPtr);

    // Set graph optimization level to ORT_ENABLE_ALL (99)
    final setOptLevel =
        api.SetSessionGraphOptimizationLevel.asFunction<
          Pointer<OrtStatus> Function(Pointer<OrtSessionOptions>, int)
        >();
    ort.checkStatus(
      setOptLevel(opts, 99),
    ); // GraphOptimizationLevel.ORT_ENABLE_ALL

    // Append each execution provider using the generic API
    final appendEP =
        api.SessionOptionsAppendExecutionProvider.asFunction<
          Pointer<OrtStatus> Function(
            Pointer<OrtSessionOptions>,
            Pointer<Char>,
            Pointer<Pointer<Char>>,
            Pointer<Pointer<Char>>,
            int,
          )
        >();

    for (final provider in providers) {
      if (provider == DartONNXExecutionProvider.cpu) {
        continue;
      }

      final providerName = provider.ortName.toNativeUtf8().cast<Char>();
      try {
        // Try to append — if the provider isn't available, ORT returns an error.
        // We silently ignore errors for non-CPU providers (they're optional).
        final status = appendEP(opts, providerName, nullptr, nullptr, 0);
        if (status != nullptr) {
          // Provider unavailable — release error and skip
          final releaseStatus =
              api.ReleaseStatus.asFunction<void Function(Pointer<OrtStatus>)>();
          releaseStatus(status);
        }
      } finally {
        calloc.free(providerName);
      }
    }

    return opts;
  }

  List<String> _getInputNames() {
    final ort = OrtFFI.instance;
    final api = ort.api.ref;

    final getCount =
        api.SessionGetInputCount.asFunction<
          Pointer<OrtStatus> Function(Pointer<OrtSession>, Pointer<Size>)
        >();
    final countPtr = calloc<Size>();
    ort.checkStatus(getCount(pointer, countPtr));
    final count = countPtr.value;
    calloc.free(countPtr);

    final getAllocator =
        api.GetAllocatorWithDefaultOptions.asFunction<
          Pointer<OrtStatus> Function(Pointer<Pointer<OrtAllocator>>)
        >();
    final allocPtr = calloc<Pointer<OrtAllocator>>();
    ort.checkStatus(getAllocator(allocPtr));
    final allocator = allocPtr.value;
    calloc.free(allocPtr);

    final getName =
        api.SessionGetInputName.asFunction<
          Pointer<OrtStatus> Function(
            Pointer<OrtSession>,
            int,
            Pointer<OrtAllocator>,
            Pointer<Pointer<Char>>,
          )
        >();

    final names = <String>[];
    for (var i = 0; i < count; i++) {
      final namePtr = calloc<Pointer<Char>>();
      ort.checkStatus(getName(pointer, i, allocator, namePtr));
      names.add(namePtr.value.cast<Utf8>().toDartString());

      // Free the name allocated by ORT
      final allocFree =
          api.AllocatorFree.asFunction<
            Pointer<OrtStatus> Function(Pointer<OrtAllocator>, Pointer<Void>)
          >();
      allocFree(allocator, namePtr.value.cast());
      calloc.free(namePtr);
    }
    return List.unmodifiable(names);
  }

  List<String> _getOutputNames() {
    final ort = OrtFFI.instance;
    final api = ort.api.ref;

    final getCount =
        api.SessionGetOutputCount.asFunction<
          Pointer<OrtStatus> Function(Pointer<OrtSession>, Pointer<Size>)
        >();
    final countPtr = calloc<Size>();
    ort.checkStatus(getCount(pointer, countPtr));
    final count = countPtr.value;
    calloc.free(countPtr);

    final getAllocator =
        api.GetAllocatorWithDefaultOptions.asFunction<
          Pointer<OrtStatus> Function(Pointer<Pointer<OrtAllocator>>)
        >();
    final allocPtr = calloc<Pointer<OrtAllocator>>();
    ort.checkStatus(getAllocator(allocPtr));
    final allocator = allocPtr.value;
    calloc.free(allocPtr);

    final getName =
        api.SessionGetOutputName.asFunction<
          Pointer<OrtStatus> Function(
            Pointer<OrtSession>,
            int,
            Pointer<OrtAllocator>,
            Pointer<Pointer<Char>>,
          )
        >();

    final names = <String>[];
    for (var i = 0; i < count; i++) {
      final namePtr = calloc<Pointer<Char>>();
      ort.checkStatus(getName(pointer, i, allocator, namePtr));
      names.add(namePtr.value.cast<Utf8>().toDartString());

      final allocFree =
          api.AllocatorFree.asFunction<
            Pointer<OrtStatus> Function(Pointer<OrtAllocator>, Pointer<Void>)
          >();
      allocFree(allocator, namePtr.value.cast());
      calloc.free(namePtr);
    }
    return List.unmodifiable(names);
  }

  @override
  String toString() =>
      'DartONNXSession(inputs: $inputNames, outputs: $outputNames)';
}
