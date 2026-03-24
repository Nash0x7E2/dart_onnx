/// Downloads the SmolLM2-135M-Instruct quantized ONNX model files plus
/// tokenizer and config files from HuggingFace into the `example/model/`
/// directory.
///
/// Run from the package root:
///   dart run tool/download_model.dart
library;

import 'dart:io';

const _baseUrl =
    'https://huggingface.co/onnx-community/SmolLM2-135M-Instruct-ONNX/resolve/main';

/// Files to download: (remotePath, localName).
const _files = [
  ('onnx/model_quantized.onnx', 'model_quantized.onnx'),
  ('config.json', 'config.json'),
  ('tokenizer.json', 'tokenizer.json'),
  ('tokenizer_config.json', 'tokenizer_config.json'),
];

void main() async {
  final scriptUri = Platform.script;
  final packageRoot = File.fromUri(scriptUri).parent.parent;
  final outputDir = Directory('${packageRoot.path}/example/model');

  if (!outputDir.existsSync()) {
    outputDir.createSync(recursive: true);
    print('Created directory: ${outputDir.path}');
  }

  final client = HttpClient();
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
  print('You can now run: dart run example/cli_chat.dart');
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
    request.followRedirects = false;
    final response = await request.close();

    if (response.statusCode >= 300 && response.statusCode < 400) {
      await response.drain<void>();
      final location = response.headers.value('location');
      if (location == null) {
        throw Exception('Redirect with no Location header from $currentUrl');
      }
      currentUrl = Uri.parse(currentUrl).resolve(location).toString();
      continue;
    }

    if (response.statusCode != 200) {
      await response.drain<void>();
      throw Exception('HTTP ${response.statusCode} downloading $currentUrl');
    }

    final sink = outputFile.openWrite();
    try {
      var downloaded = 0;
      var lastPrint = 0;
      await for (final chunk in response) {
        sink.add(chunk);
        downloaded += chunk.length;
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
