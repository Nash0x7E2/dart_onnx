import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

import 'ffi/ort_bindings.dart';
import 'ort_ffi.dart';

/// A Dart-native representation of an ONNX tensor.
///
/// Handles the translation between Dart's [TypedData] arrays
/// (e.g. [Float32List], [Int64List]) and native C memory.
///
/// Memory is automatically freed when the object is garbage collected
/// (via [NativeFinalizer]), but you may call [dispose] for deterministic cleanup.
///
/// ```dart
/// final tensor = DartONNXTensor.float32(
///   data: Float32List.fromList([1.0, 2.0, 3.0]),
///   shape: [1, 3],
/// );
/// ```
class DartONNXTensor implements Finalizable {
  static final _finalizer = NativeFinalizer(_releaseFn);

  static final Pointer<NativeFunction<Void Function(Pointer<Void>)>>
  _releaseFn = OrtFFI.instance.api.ref.ReleaseValue.cast();

  /// Raw pointer to the OrtValue representing this tensor.
  final Pointer<OrtValue> _ptr;

  /// The shape of this tensor.
  final List<int> shape;

  /// The element data type of this tensor.
  final ONNXTensorElementDataType elementType;

  bool _disposed = false;

  DartONNXTensor._(this._ptr, this.shape, this.elementType) {
    _finalizer.attach(this, _ptr.cast(), detach: this);
  }

  /// Get the raw OrtValue pointer. Throws if already disposed.
  Pointer<OrtValue> get pointer {
    if (_disposed) {
      throw StateError('DartONNXTensor has been disposed.');
    }
    return _ptr;
  }

  /// Create a Float32 tensor from a [Float32List] and a shape.
  factory DartONNXTensor.float32({
    required Float32List data,
    required List<int> shape,
  }) {
    return _createTensor(
      data.buffer.asUint8List(),
      shape,
      ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    );
  }

  /// Create a Float64 (Double) tensor.
  factory DartONNXTensor.float64({
    required Float64List data,
    required List<int> shape,
  }) {
    return _createTensor(
      data.buffer.asUint8List(),
      shape,
      ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
    );
  }

  /// Create an Int32 tensor.
  factory DartONNXTensor.int32({
    required Int32List data,
    required List<int> shape,
  }) {
    return _createTensor(
      data.buffer.asUint8List(),
      shape,
      ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
    );
  }

  /// Create an Int64 tensor.
  factory DartONNXTensor.int64({
    required Int64List data,
    required List<int> shape,
  }) {
    return _createTensor(
      data.buffer.asUint8List(),
      shape,
      ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    );
  }

  /// Create a Uint8 tensor.
  factory DartONNXTensor.uint8({
    required Uint8List data,
    required List<int> shape,
  }) {
    return _createTensor(
      data,
      shape,
      ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
    );
  }

  /// Internal method to create an OrtValue tensor from raw bytes.
  static DartONNXTensor _createTensor(
    Uint8List rawBytes,
    List<int> shape,
    ONNXTensorElementDataType elementType,
  ) {
    final ort = OrtFFI.instance;
    final api = ort.api.ref;

    // 1. Create CPU MemoryInfo
    final createCpuMemoryInfo =
        api.CreateCpuMemoryInfo.asFunction<
          Pointer<OrtStatus> Function(int, int, Pointer<Pointer<OrtMemoryInfo>>)
        >();
    final memoryInfoPtr = calloc<Pointer<OrtMemoryInfo>>();

    // OrtAllocatorType.OrtArenaAllocator = 0, OrtMemType.OrtMemTypeDefault = 0
    final memStatus = createCpuMemoryInfo(
      1,
      0,
      memoryInfoPtr,
    ); // 1 = OrtDeviceAllocator
    ort.checkStatus(memStatus);

    // 2. Allocate native memory and copy data
    final dataPtr = calloc<Uint8>(rawBytes.length);
    dataPtr.asTypedList(rawBytes.length).setAll(0, rawBytes);

    // 3. Create shape array
    final shapePtr = calloc<Int64>(shape.length);
    for (var i = 0; i < shape.length; i++) {
      shapePtr[i] = shape[i];
    }

    // 4. Create the tensor
    final createTensor =
        api.CreateTensorWithDataAsOrtValue.asFunction<
          Pointer<OrtStatus> Function(
            Pointer<OrtMemoryInfo>,
            Pointer<Void>,
            int,
            Pointer<Int64>,
            int,
            int,
            Pointer<Pointer<OrtValue>>,
          )
        >();

    final outPtr = calloc<Pointer<OrtValue>>();
    try {
      final status = createTensor(
        memoryInfoPtr.value,
        dataPtr.cast(),
        rawBytes.length,
        shapePtr,
        shape.length,
        elementType.value,
        outPtr,
      );
      ort.checkStatus(status);

      return DartONNXTensor._(
        outPtr.value,
        List.unmodifiable(shape),
        elementType,
      );
    } finally {
      // Release memory info (tensor now owns a reference)
      final releaseMemInfo =
          api.ReleaseMemoryInfo.asFunction<
            void Function(Pointer<OrtMemoryInfo>)
          >();
      releaseMemInfo(memoryInfoPtr.value);
      calloc.free(memoryInfoPtr);
      calloc.free(shapePtr);
      // Note: dataPtr must NOT be freed here — the OrtValue references it.
      // It will be valid as long as the OrtValue exists.
      // We intentionally leak dataPtr; it's freed when the OrtValue is released.
      // TODO: track dataPtr and free in dispose/finalizer
    }
  }

  /// Create a [DartONNXTensor] from a raw [Pointer<OrtValue>] (used internally
  /// for output tensors from session.run).
  factory DartONNXTensor.fromOrtValue(Pointer<OrtValue> ptr) {
    final ort = OrtFFI.instance;
    final api = ort.api.ref;

    // Get tensor type and shape info
    final getTypeAndShape =
        api.GetTensorTypeAndShape.asFunction<
          Pointer<OrtStatus> Function(
            Pointer<OrtValue>,
            Pointer<Pointer<OrtTensorTypeAndShapeInfo>>,
          )
        >();
    final infoPtr = calloc<Pointer<OrtTensorTypeAndShapeInfo>>();
    ort.checkStatus(getTypeAndShape(ptr, infoPtr));

    // Get element type
    final getElementType =
        api.GetTensorElementType.asFunction<
          Pointer<OrtStatus> Function(
            Pointer<OrtTensorTypeAndShapeInfo>,
            Pointer<UnsignedInt>,
          )
        >();
    final typePtr = calloc<UnsignedInt>();
    ort.checkStatus(getElementType(infoPtr.value, typePtr));
    final elementType = ONNXTensorElementDataType.fromValue(typePtr.value);

    // Get dimensions
    final getDimCount =
        api.GetDimensionsCount.asFunction<
          Pointer<OrtStatus> Function(
            Pointer<OrtTensorTypeAndShapeInfo>,
            Pointer<Size>,
          )
        >();
    final dimCountPtr = calloc<Size>();
    ort.checkStatus(getDimCount(infoPtr.value, dimCountPtr));
    final dimCount = dimCountPtr.value;

    final getDims =
        api.GetDimensions.asFunction<
          Pointer<OrtStatus> Function(
            Pointer<OrtTensorTypeAndShapeInfo>,
            Pointer<Int64>,
            int,
          )
        >();
    final dimsPtr = calloc<Int64>(dimCount);
    ort.checkStatus(getDims(infoPtr.value, dimsPtr, dimCount));

    final shape = <int>[];
    for (var i = 0; i < dimCount; i++) {
      shape.add(dimsPtr[i]);
    }

    // Release the type+shape info
    final releaseInfo =
        api.ReleaseTensorTypeAndShapeInfo.asFunction<
          void Function(Pointer<OrtTensorTypeAndShapeInfo>)
        >();
    releaseInfo(infoPtr.value);

    calloc.free(infoPtr);
    calloc.free(typePtr);
    calloc.free(dimCountPtr);
    calloc.free(dimsPtr);

    return DartONNXTensor._(ptr, List.unmodifiable(shape), elementType);
  }

  /// Get the tensor data as a Dart [TypedData] list.
  ///
  /// Returns the appropriate type based on [elementType]:
  /// - Float → [Float32List]
  /// - Double → [Float64List]
  /// - Int32 → [Int32List]
  /// - Int64 → [Int64List]
  /// - Uint8 → [Uint8List]
  TypedData get data {
    final ort = OrtFFI.instance;
    final api = ort.api.ref;

    final getData =
        api.GetTensorMutableData.asFunction<
          Pointer<OrtStatus> Function(Pointer<OrtValue>, Pointer<Pointer<Void>>)
        >();

    final dataPtr = calloc<Pointer<Void>>();
    try {
      ort.checkStatus(getData(pointer, dataPtr));

      final totalElements = shape.fold<int>(1, (a, b) => a * b);
      final raw = dataPtr.value;

      switch (elementType) {
        case ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
          return raw.cast<Float>().asTypedList(totalElements);
        case ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
          return raw.cast<Double>().asTypedList(totalElements);
        case ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
          return raw.cast<Int32>().asTypedList(totalElements);
        case ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
          return raw.cast<Int64>().asTypedList(totalElements);
        case ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
          return raw.cast<Uint8>().asTypedList(totalElements);
        default:
          throw UnsupportedError(
            'Unsupported tensor element type: $elementType',
          );
      }
    } finally {
      calloc.free(dataPtr);
    }
  }

  /// Manually dispose the underlying native memory.
  ///
  /// Optional — [NativeFinalizer] handles this automatically.
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _finalizer.detach(this);
    final release = OrtFFI.instance.api.ref.ReleaseValue
        .asFunction<void Function(Pointer<OrtValue>)>();
    release(_ptr);
  }

  @override
  String toString() =>
      'DartONNXTensor(shape: $shape, elementType: $elementType)';
}
