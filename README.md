# dart_inference

A Dart monorepo for high-performance AI inference and ML integrations. 

Currently, it contains bindings to ONNX Runtime via FFI, with future plans to add high-level Chat APIs and abstractions for interacting with various models (including LLMs) directly in Dart.

## Packages

This repository contains the following packages:

- [`dart_onnx`](./packages/dart_onnx): A cross-platform Dart package for running ONNX models using ONNX Runtime via Dart FFI.
*(More packages, such as a high-level chat API, will be added here as they are developed.)*

## Getting Started

Because this is a monorepo managed with [Melos](https://melos.invertase.dev/), you can run the following commands to get started:

### Prerequisites

You will need the Melos CLI installed globally:

```bash
dart pub global activate melos
```

### Initial Setup

To link local packages together and install all dependencies, run the following command from the root of the repository:

```bash
melos bootstrap
```

### Useful Commands

- `melos run analyze`: Runs `dart analyze` across all packages.
- `melos run format`: Formats all Dart code.
- `melos run test`: Runs tests in all packages.

For instructions on using a specific package (e.g., `dart_onnx`), see the `README.md` inside its respective folder.
