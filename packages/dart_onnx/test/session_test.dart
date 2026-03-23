import 'dart:typed_data';
import 'package:dart_onnx/dart_onnx.dart';
import 'package:test/test.dart';
import 'dart:io';

void main() {
  group('DartONNX Env', () {
    test('Initialization allows one environment', () {
      final env = DartONNX();
      expect(env, isNotNull);
      env.dispose();
    });

    test('dispose() is idempotent', () {
      final env = DartONNX();
      env.dispose();
      env.dispose(); // Should not throw
    });

    test('Calling pointer after dispose throws StateError', () {
      final env = DartONNX();
      env.dispose();
      expect(() => env.pointer, throwsStateError);
    });
  });

  group('DartONNXSession Init', () {
    late DartONNX env;

    setUpAll(() {
      env = DartONNX();
    });

    tearDownAll(() {
      env.dispose();
    });

    test('Fails on incorrect file path', () {
      expect(
        () => DartONNXSession.fromFile(env, 'does_not_exist.onnx'),
        throwsA(isA<DartONNXException>()),
      );
    });

    test('Initializes with generated test model', () {
      final session = DartONNXSession.fromFile(
        env,
        'test/assets/models/identity.onnx',
      );
      expect(session.inputNames.length, 1);
      expect(session.outputNames.length, 1);

      // The identity test has input ['X']
      expect(session.inputNames.first, 'X');
      expect(session.outputNames.first, 'Y');

      session.dispose();
    });

    test('Initializes via byte array', () {
      final bytes = File('test/assets/models/add.onnx').readAsBytesSync();
      final session = DartONNXSession.fromBytes(env, bytes);

      expect(session.inputNames.length, 2); // 'A' and 'B'
      expect(session.outputNames.length, 1); // 'C'

      expect(session.inputNames.contains('A'), isTrue);
      expect(session.inputNames.contains('B'), isTrue);
      expect(session.outputNames.contains('C'), isTrue);

      session.dispose();
    });

    test('run() throws ArgumentError on unknown input name', () {
      final session = DartONNXSession.fromFile(
        env,
        'test/assets/models/identity.onnx',
      );

      final tensor = DartONNXTensor.float32(
        data: Float32List.fromList([1.0]),
        shape: [1],
      );

      expect(() => session.run({'UNKNOWN_INPUT': tensor}), throwsArgumentError);

      tensor.dispose();
      session.dispose();
    });
  });

  group('DartONNXSession — Use-After-Dispose', () {
    late DartONNX env;

    setUpAll(() {
      env = DartONNX();
    });

    tearDownAll(() {
      env.dispose();
    });

    test('Calling pointer after dispose throws StateError', () {
      final session = DartONNXSession.fromFile(
        env,
        'test/assets/models/identity.onnx',
      );
      session.dispose();
      expect(() => session.pointer, throwsStateError);
    });

    test('dispose() is idempotent', () {
      final session = DartONNXSession.fromFile(
        env,
        'test/assets/models/identity.onnx',
      );
      session.dispose();
      session.dispose(); // Should not throw
    });
  });
}
