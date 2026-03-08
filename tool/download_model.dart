/// Downloads the SmolLM2-135M quantized ONNX model files from HuggingFace
/// into the `example/onnx/` directory.
///
/// Run from the package root:
///   dart run tool/download_model.dart
library;

import 'dart:io';

const _baseUrl =
    'https://huggingface.co/onnx-community/SmolLM2-135M-ONNX/resolve/main';

/// Files to download. Each entry is (remotePath, localPath relative to outputDir).
const _files = [
  ('onnx/model_quantized.onnx', 'model_quantized.onnx'),
  ('onnx/model_quantized.onnx_data', 'model_quantized.onnx_data'),
];

void main() async {
  // Output directory: example/onnx/ relative to the package root.
  final scriptUri = Platform.script;
  final packageRoot = File.fromUri(scriptUri).parent.parent;
  final outputDir = Directory('${packageRoot.path}/example/onnx');

  if (!outputDir.existsSync()) {
    outputDir.createSync(recursive: true);
    print('Created directory: ${outputDir.path}');
  }

  final client = HttpClient();
  // HuggingFace redirects to CDN — follow redirects automatically.
  client.maxConnectionsPerHost = 2;

  try {
    for (final (remotePath, localName) in _files) {
      final outputFile = File('${outputDir.path}/$localName');

      if (outputFile.existsSync()) {
        print('✓ Already exists, skipping: $localName');
        continue;
      }

      final url = '$_baseUrl/$remotePath';
      print('⬇  Downloading $localName ...');
      print('   from: $url');

      await _downloadWithRedirects(client, url, outputFile);

      final sizeBytes = outputFile.lengthSync();
      final sizeMb = (sizeBytes / 1024 / 1024).toStringAsFixed(1);
      print('✓ Saved $localName ($sizeMb MB)');
    }
  } finally {
    client.close();
  }

  print('\nDone! Model files are at: ${outputDir.path}');
  print('You can now run: dart run example/dart_onnx_example.dart');
}

/// Downloads [url] to [outputFile], following HTTP redirects manually.
Future<void> _downloadWithRedirects(
  HttpClient client,
  String url,
  File outputFile,
) async {
  const maxRedirects = 10;
  var currentUrl = url;

  for (var i = 0; i < maxRedirects; i++) {
    final uri = Uri.parse(currentUrl);
    final request = await client.getUrl(uri);
    // Disable automatic redirect following so we can handle 302 → follow HEAD.
    request.followRedirects = false;
    final response = await request.close();

    if (response.statusCode >= 300 && response.statusCode < 400) {
      // Drain response body and follow redirect.
      await response.drain<void>();
      final location = response.headers.value('location');
      if (location == null) {
        throw Exception('Redirect with no Location header from $currentUrl');
      }
      // Location may be relative or absolute.
      currentUrl = Uri.parse(currentUrl).resolve(location).toString();
      continue;
    }

    if (response.statusCode != 200) {
      await response.drain<void>();
      throw Exception('HTTP ${response.statusCode} downloading $currentUrl');
    }

    // Stream to file.
    final sink = outputFile.openWrite();
    try {
      var downloaded = 0;
      var lastPrint = 0;
      await for (final chunk in response) {
        sink.add(chunk);
        downloaded += chunk.length;
        // Print progress every ~10 MB.
        if (downloaded - lastPrint >= 10 * 1024 * 1024) {
          final mb = (downloaded / 1024 / 1024).toStringAsFixed(0);
          stdout.write('\r   $mb MB downloaded...');
          lastPrint = downloaded;
        }
      }
      if (downloaded > 10 * 1024 * 1024) stdout.writeln();
    } finally {
      await sink.close();
    }
    return;
  }

  throw Exception('Too many redirects for $url');
}
