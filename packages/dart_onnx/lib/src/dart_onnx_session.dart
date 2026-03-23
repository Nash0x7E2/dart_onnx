import 'dart:ffi';
import 'dart:io';
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
    return using((Arena arena) {
      final ort = OrtFFI.instance;
      final api = ort.api.ref;

      final sessionOptions = _createSessionOptions(ort, executionProviders);
      final pathNative = Platform.isWindows
          ? modelPath.toNativeUtf16(allocator: arena).cast<Char>()
          : modelPath.toNativeUtf8(allocator: arena).cast<Char>();
      final outPtr = arena<Pointer<OrtSession>>();

      try {
        final createSession =
            api.CreateSession.asFunction<
              Pointer<OrtStatus> Function(
                Pointer<OrtEnv>,
                Pointer<Char>,
                Pointer<OrtSessionOptions>,
                Pointer<Pointer<OrtSession>>,
              )
            >();
        final status = createSession(
          env.pointer,
          pathNative,
          sessionOptions,
          outPtr,
        );
        ort.checkStatus(status);
        return DartONNXSession._(outPtr.value);
      } finally {
        final releaseOpts =
            api.ReleaseSessionOptions.asFunction<
              void Function(Pointer<OrtSessionOptions>)
            >();
        releaseOpts(sessionOptions);
      }
    });
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
    return using((Arena arena) {
      final ort = OrtFFI.instance;
      final api = ort.api.ref;

      final sessionOptions = _createSessionOptions(ort, executionProviders);
      final modelData = arena<Uint8>(modelBytes.length);
      for (var i = 0; i < modelBytes.length; i++) {
        modelData[i] = modelBytes[i];
      }
      final outPtr = arena<Pointer<OrtSession>>();

      try {
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
        final releaseOpts =
            api.ReleaseSessionOptions.asFunction<
              void Function(Pointer<OrtSessionOptions>)
            >();
        releaseOpts(sessionOptions);
      }
    });
  }

  /// Run inference on the model.
  ///
  /// [inputs] maps input names to their [DartONNXTensor] values.
  /// Returns a map of output names to their result [DartONNXTensor] values.
  Map<String, DartONNXTensor> run(Map<String, DartONNXTensor> inputs) {
    for (final name in inputs.keys) {
      if (!inputNames.contains(name)) {
        throw ArgumentError(
          'Unknown input name "$name". Expected one of: $inputNames',
        );
      }
    }

    return using((Arena arena) {
      final ort = OrtFFI.instance;
      final api = ort.api.ref;

      final inputCount = inputs.length;
      final outputCount = outputNames.length;

      // Prepare input names
      final inputNamesPtr = arena<Pointer<Char>>(inputCount);
      var i = 0;
      for (final name in inputs.keys) {
        inputNamesPtr[i] = name.toNativeUtf8(allocator: arena).cast();
        i++;
      }

      // Prepare input values
      final inputValuesPtr = arena<Pointer<OrtValue>>(inputCount);
      i = 0;
      for (final tensor in inputs.values) {
        inputValuesPtr[i] = tensor.pointer;
        i++;
      }

      // Prepare output names
      final outputNamesPtr = arena<Pointer<Char>>(outputCount);
      for (var j = 0; j < outputCount; j++) {
        outputNamesPtr[j] = outputNames[j]
            .toNativeUtf8(allocator: arena)
            .cast();
      }

      // Prepare output values array (ORT fills this)
      final outputValuesPtr = arena<Pointer<OrtValue>>(outputCount);

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
    });
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

  /// Create an OrtSessionOptions with the given execution providers.
  static Pointer<OrtSessionOptions> _createSessionOptions(
    OrtFFI ort,
    List<DartONNXExecutionProvider> providers,
  ) {
    final api = ort.api.ref;

    return using((Arena arena) {
      final createOpts =
          api.CreateSessionOptions.asFunction<
            Pointer<OrtStatus> Function(Pointer<Pointer<OrtSessionOptions>>)
          >();
      final optsPtr = arena<Pointer<OrtSessionOptions>>();
      ort.checkStatus(createOpts(optsPtr));
      final opts = optsPtr.value;

      final setOptLevel =
          api.SetSessionGraphOptimizationLevel.asFunction<
            Pointer<OrtStatus> Function(Pointer<OrtSessionOptions>, int)
          >();
      ort.checkStatus(setOptLevel(opts, 99)); // ORT_ENABLE_ALL

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
        if (provider == DartONNXExecutionProvider.cpu) continue;

        final providerName = provider.ortName
            .toNativeUtf8(allocator: arena)
            .cast<Char>();
        final status = appendEP(opts, providerName, nullptr, nullptr, 0);
        if (status != nullptr) {
          final releaseStatus =
              api.ReleaseStatus.asFunction<void Function(Pointer<OrtStatus>)>();
          releaseStatus(status);
        }
      }

      return opts;
    });
  }

  List<String> _getInputNames() => _getNames(isInput: true);
  List<String> _getOutputNames() => _getNames(isInput: false);

  List<String> _getNames({required bool isInput}) {
    return using((Arena arena) {
      final ort = OrtFFI.instance;
      final api = ort.api.ref;

      final getCount = isInput
          ? api.SessionGetInputCount.asFunction<
              Pointer<OrtStatus> Function(Pointer<OrtSession>, Pointer<Size>)
            >()
          : api.SessionGetOutputCount.asFunction<
              Pointer<OrtStatus> Function(Pointer<OrtSession>, Pointer<Size>)
            >();

      final countPtr = arena<Size>();
      ort.checkStatus(getCount(pointer, countPtr));
      final count = countPtr.value;

      final getAllocator =
          api.GetAllocatorWithDefaultOptions.asFunction<
            Pointer<OrtStatus> Function(Pointer<Pointer<OrtAllocator>>)
          >();
      final allocPtr = arena<Pointer<OrtAllocator>>();
      ort.checkStatus(getAllocator(allocPtr));
      final allocator = allocPtr.value;

      final getName = isInput
          ? api.SessionGetInputName.asFunction<
              Pointer<OrtStatus> Function(
                Pointer<OrtSession>,
                int,
                Pointer<OrtAllocator>,
                Pointer<Pointer<Char>>,
              )
            >()
          : api.SessionGetOutputName.asFunction<
              Pointer<OrtStatus> Function(
                Pointer<OrtSession>,
                int,
                Pointer<OrtAllocator>,
                Pointer<Pointer<Char>>,
              )
            >();

      final names = <String>[];
      for (var i = 0; i < count; i++) {
        final namePtr = arena<Pointer<Char>>();
        ort.checkStatus(getName(pointer, i, allocator, namePtr));
        names.add(namePtr.value.cast<Utf8>().toDartString());

        final allocFree =
            api.AllocatorFree.asFunction<
              Pointer<OrtStatus> Function(Pointer<OrtAllocator>, Pointer<Void>)
            >();
        allocFree(allocator, namePtr.value.cast());
      }
      return List.unmodifiable(names);
    });
  }

  @override
  String toString() =>
      'DartONNXSession(inputs: $inputNames, outputs: $outputNames)';
}
