# dart_onnx

A cross-platform Dart package for running ONNX models using ONNX Runtime via Dart FFI.
This package provides a high-level, Dart-idiomatic API for performing inference with ONNX models without writing C/C++ code.

## Installation

Add `dart_onnx` to your `pubspec.yaml`:

```yaml
dependencies:
  dart_onnx: ^0.1.0
```

Or install it via Dart or Flutter CLI:

```bash
dart pub add dart_onnx
# or
flutter pub add dart_onnx
```

## Setup

The `dart_onnx` package uses Dart FFI to bind to the ONNX Runtime C library (`libonnxruntime`). 
You must install the ONNX Runtime dynamic library on your system before running models.

### macOS

Install via Homebrew:

```bash
brew install onnxruntime
```

If Dart cannot find the library automatically (e.g., throwing `Failed to load dynamic library 'libonnxruntime.dylib'`), you can supply the path to the library by setting the `DART_ONNX_LIB_PATH` environment variable:

```bash
export DART_ONNX_LIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib
```

### Linux

Download the appropriate ONNX Runtime release from the [GitHub releases page](https://github.com/microsoft/onnxruntime/releases), extract it, and place the `.so` library in your library path, or set `DART_ONNX_LIB_PATH` to point directly to it:

```bash
export DART_ONNX_LIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so
```

### Windows

Download the appropriate Windows ONNX Runtime release, extract it, and add the directory containing `onnxruntime.dll` to your `PATH`, or set `DART_ONNX_LIB_PATH`:

```cmd
set DART_ONNX_LIB_PATH=C:\path\to\onnxruntime\lib\onnxruntime.dll
```

## Basic Usage

Below is a minimal example of how to initialize the ONNX Runtime, load a model, and perform inference. 
For a complete, workable example (like running a language model), see the `/example` directory.

```dart
import 'dart:typed_data';
import 'package:dart_onnx/dart_onnx.dart';

void main() {
  // 1. Initialize the ONNX Runtime environment
  final env = DartONNX(loggingLevel: DartONNXLoggingLevel.warning);
  print('ONNX Runtime Version: ${env.ortVersion}');

  // 2. Load the model session
  // You can optionally specify execution providers such as CoreML or CUDA.
  final session = DartONNXSession.fromFile(
    env,
    'path/to/your/model.onnx',
    executionProviders: [
      DartONNXExecutionProvider.coreML, // Uses Apple Neural Engine if available
      DartONNXExecutionProvider.cpu,    // Fallback
    ],
  );

  // 3. Prepare input tensors
  // Ensure the shape and data type match your model's expected inputs!
  final inputData = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);
  final inputTensor = DartONNXTensor.float32(
    data: inputData,
    shape: [1, 4], // Example shape: batch size 1, 4 features
  );

  // 4. Run inference
  // Pass a map of input names to tensors.
  final inputs = {'input_name': inputTensor};
  final outputs = session.run(inputs);

  // 5. Read output
  final outputTensor = outputs['output_name'];
  if (outputTensor != null) {
    final outputData = outputTensor.data as Float32List;
    print('Output data: $outputData');
  }

  // 6. Cleanup to prevent memory leaks
  inputTensor.dispose();
  for (final tensor in outputs.values) {
    tensor.dispose();
  }
  session.dispose();
}
```

## Additional Information

### Generating Test Models

The tiny ONNX models used for unit testing (`identity.onnx`, `add.onnx`, `multi_output.onnx`) are checked into the repository under `test/assets/models/`. If you need to regenerate them or add new ones, we provide a Python script:

```bash
# From the project root, ensure you have python with the 'onnx' package installed
# (e.g. via `pip install onnx` or `uv run --with onnx python3 tool/generate_test_models.py`)
python3 tool/generate_test_models.py
```

For more details on ONNX Runtime capabilities, refer to the [ONNX Runtime documentation](https://onnxruntime.ai/docs/). Feel free to open issues or contribute to this package in the [repository](https://github.com/nash/dart_onnx).
