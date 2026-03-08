import 'dart:ffi';
import 'package:ffi/ffi.dart';

import 'ffi/ort_bindings.dart';
import 'ort_ffi.dart';
import 'dart_onnx_logging_level.dart';

/// The ONNX Runtime environment.
///
/// Represents a global ORT environment that manages thread pools and
/// logging. You typically need one of these per application.
///
/// ```dart
/// final env = DartONNX(loggingLevel: DartONNXLoggingLevel.warning);
/// // ... use env to create sessions ...
/// env.dispose(); // optional — GC will clean up via NativeFinalizer
/// ```
class DartONNX implements Finalizable {
  static final _finalizer = NativeFinalizer(_releaseFn);

  static final Pointer<NativeFunction<Void Function(Pointer<Void>)>>
  _releaseFn = OrtFFI.instance.api.ref.ReleaseEnv.cast();

  /// Raw pointer to the OrtEnv. Exposed for internal use.
  final Pointer<OrtEnv> _ptr;
  bool _disposed = false;

  /// The logging level this environment was created with.
  final DartONNXLoggingLevel loggingLevel;

  /// Create a new ONNX Runtime environment.
  ///
  /// [loggingLevel] controls the ORT internal logging verbosity.
  /// [logId] is an optional identifier for the logging output.
  factory DartONNX({
    DartONNXLoggingLevel loggingLevel = DartONNXLoggingLevel.warning,
    String logId = 'dart_onnx',
  }) {
    final ort = OrtFFI.instance;
    final createEnv = ort.api.ref.CreateEnv
        .asFunction<
          Pointer<OrtStatus> Function(
            int,
            Pointer<Char>,
            Pointer<Pointer<OrtEnv>>,
          )
        >();

    final logIdNative = logId.toNativeUtf8().cast<Char>();
    final outPtr = calloc<Pointer<OrtEnv>>();
    try {
      final status = createEnv(loggingLevel.value, logIdNative, outPtr);
      ort.checkStatus(status);
      final env = DartONNX._(outPtr.value, loggingLevel);
      return env;
    } finally {
      calloc.free(logIdNative);
      calloc.free(outPtr);
    }
  }

  DartONNX._(this._ptr, this.loggingLevel) {
    _finalizer.attach(this, _ptr.cast(), detach: this);
  }

  /// Get the raw OrtEnv pointer. Throws if already disposed.
  Pointer<OrtEnv> get pointer {
    if (_disposed) {
      throw StateError('DartONNX environment has been disposed.');
    }
    return _ptr;
  }

  /// Manually release the underlying native resources.
  ///
  /// This is optional — Dart's garbage collector will release the
  /// environment automatically via [NativeFinalizer]. Call this if you
  /// need deterministic cleanup.
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _finalizer.detach(this);
    final release = OrtFFI.instance.api.ref.ReleaseEnv
        .asFunction<void Function(Pointer<OrtEnv>)>();
    release(_ptr);
  }

  /// Get the ONNX Runtime version string.
  String get ortVersion {
    final getVersionString = OrtFFI.instance.bindings
        .OrtGetApiBase()
        .ref
        .GetVersionString
        .asFunction<Pointer<Char> Function()>();
    return getVersionString().cast<Utf8>().toDartString();
  }
}
