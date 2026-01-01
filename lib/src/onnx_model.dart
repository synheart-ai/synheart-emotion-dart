import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:path_provider/path_provider.dart';

import 'emotion_error.dart';

/// ONNX-based model loader for emotion inference.
///
/// Loads and runs ONNX models with metadata from accompanying meta.json
/// files.
class OnnxEmotionModel {
  OnnxEmotionModel._({
    required this.modelId,
    required this.inputNames,
    required this.classNames,
    required OrtSession session,
    required Map<String, dynamic> metadata,
  }) : _session = session,
       _metadata = metadata;

  final String modelId;
  final List<String> inputNames;
  final List<String> classNames;
  final OrtSession _session;
  final Map<String, dynamic> _metadata;

  /// Load ONNX model from assets
  ///
  /// Automatically detects metadata file based on model path pattern:
  /// - ExtraTrees_60_5.onnx -> ExtraTrees_metadata_60_5.json
  /// - ExtraTrees_60_5_nozipmap.onnx -> ExtraTrees_metadata_60_5_nozipmap.json
  static Future<OnnxEmotionModel> loadFromAsset({
    required String modelAssetPath,
    String? metaAssetPath,
  }) async {
    // Auto-detect metadata path if not provided
    var resolvedMetaAssetPath = metaAssetPath;
    if (resolvedMetaAssetPath == null) {
      final fileName = modelAssetPath.split('/').last;
      if (fileName.contains('_')) {
        // Pattern: ExtraTrees_60_5_nozipmap.onnx -> ExtraTrees_metadata_60_5_nozipmap.json
        final parts = fileName.replaceAll('.onnx', '').split('_');
        if (parts.length >= 4) {
          // Reconstruct metadata filename
          final metaFileName =
              '${parts[0]}_metadata_${parts[1]}_${parts[2]}_${parts.sublist(3).join('_')}.json';
          resolvedMetaAssetPath = modelAssetPath.replaceAll(
            fileName,
            metaFileName,
          );
        } else {
          // Fallback: try simple replacement
          resolvedMetaAssetPath = modelAssetPath.replaceAll(
            '.onnx',
            '_metadata.json',
          );
        }
      } else {
        resolvedMetaAssetPath = modelAssetPath.replaceAll(
          '.onnx',
          '_metadata.json',
        );
      }
    }
    try {
      // Load metadata
      final metaString = await rootBundle.loadString(resolvedMetaAssetPath);
      final metadata = json.decode(metaString) as Map<String, dynamic>;

      final modelId = metadata['model_id'] as String;
      final schema = metadata['schema'] as Map<String, dynamic>;
      final inputNames = List<String>.from(schema['input_names'] as List);
      final output = metadata['output'] as Map<String, dynamic>;
      final classNames = List<String>.from(output['class_names'] as List);

      // Load ONNX model
      final onnxRuntime = OnnxRuntime();
      OrtSession session;

      try {
        if (kIsWeb) {
          // For web, load directly from assets
          session = await onnxRuntime.createSessionFromAsset(modelAssetPath);
        } else {
          // For native platforms, extract asset to temporary file
          final directory = await getTemporaryDirectory();
          final fileName = modelAssetPath.split('/').last;
          final modelPath =
              '${directory.path}${Platform.pathSeparator}$fileName';

          final file = File(modelPath);
          if (!file.existsSync()) {
            final byteData = await rootBundle.load(modelAssetPath);
            // ignore: avoid_slow_async_io
            await file.writeAsBytes(byteData.buffer.asUint8List());
          }

          session = await onnxRuntime.createSession(modelPath);
        }
      } catch (e) {
        throw EmotionError.badInput('Failed to load ONNX model: $e');
      }

      return OnnxEmotionModel._(
        modelId: modelId,
        inputNames: inputNames,
        classNames: classNames,
        session: session,
        metadata: metadata,
      );
    } catch (e) {
      throw EmotionError.badInput('Failed to load ONNX model from asset: $e');
    }
  }

  /// Predict emotion probabilities from features
  ///
  /// Note: This throws an error since ONNX inference is async.
  /// Use predictAsync() instead.
  Map<String, double> predict(Map<String, double> features) {
    throw EmotionError.badInput(
      'ONNX inference requires async. Use predictAsync() instead.',
    );
  }

  /// Predict emotion probabilities from features (async version)
  Future<Map<String, double>> predictAsync(Map<String, double> features) async {
    try {
      // Extract feature vector in the order expected by the model
      final inputVector = _extractFeatureVector(features);

      // Prepare input data as Float32List
      final inputData = Float32List.fromList(inputVector);
      final inputShape = [1, inputNames.length];

      // Create input tensor
      final inputTensor = await OrtValue.fromList(inputData, inputShape);

      // Get input name from session or use default
      final inputName =
          _session.inputNames.isNotEmpty
              ? _session.inputNames.first
              : 'float_input';

      // Run inference
      final inputs = {inputName: inputTensor};
      final outputs = await _session.run(inputs);

      // ExtraTrees ONNX models output: [label, probabilities]
      // Models converted without ZipMap output tensor probabilities directly
      List<double> probabilities;

      try {
        final outputKeys = outputs.keys.toList();

        if (outputKeys.length < 2) {
          throw EmotionError.badInput(
            'Expected 2 outputs (label, probabilities), got ${outputKeys.length}',
          );
        }

        // Identify outputs by name (order can vary)
        String? probKey;

        for (final key in outputKeys) {
          final lowerKey = key.toLowerCase();
          if (lowerKey.contains('prob')) {
            probKey = key;
          }
        }

        // Fallback: if not found by name, use position (first = label, second = probabilities)
        // This matches Python: outputs[0] is label, outputs[1] is probabilities
        final resolvedProbKey = probKey ?? outputKeys[1];

        final probValue = outputs[resolvedProbKey];
        if (probValue == null) {
          throw EmotionError.badInput(
            'Could not find probabilities output "$resolvedProbKey"',
          );
        }

        // Extract probabilities - handle shape [1, num_classes] like [[0.3, 0.7]]
        final List<dynamic> probData = await probValue.asList();
        List<dynamic> probList = probData;

        // Handle shape [1, num_classes] - take first element to get [num_classes]
        // If probData is [[0.3, 0.7]], extract [0] to get [0.3, 0.7]
        if (probList.isNotEmpty && probList.first is List) {
          // It's nested: [[0.3, 0.7]] -> take [0] to get [0.3, 0.7]
          probList = probList.first as List;
        }

        // Now probData should be a 1D list like [0.3, 0.7]
        if (probList.length != classNames.length) {
          throw EmotionError.badInput(
            'Invalid probability shape: length ${probList.length}, expected ${classNames.length}',
          );
        }

        probabilities = <double>[];
        for (int i = 0; i < classNames.length; i++) {
          final val = probList[i];
          if (val is num) {
            probabilities.add(val.toDouble());
          } else {
            throw EmotionError.badInput(
              'Invalid probability value at index $i: $val',
            );
          }
        }

        // Normalize (same as Python: probs = probs / (prob_sum + 1e-8))
        final sum = probabilities.fold(0.0, (a, b) => a + b);
        if (sum > 0 && (sum - 1.0).abs() > 0.001) {
          // Only normalize if not already close to 1.0 (within tolerance)
          for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = probabilities[i] / (sum + 1e-8);
          }
        }
      } catch (e) {
        if (e is EmotionError) {
          rethrow;
        }
        throw EmotionError.badInput('ONNX inference failed: $e');
      }

      // Release resources
      await inputTensor.dispose();
      // Output tensors are automatically managed

      // Convert to map with class names
      final result = <String, double>{};
      for (var i = 0; i < classNames.length && i < probabilities.length; i++) {
        result[classNames[i]] = probabilities[i];
      }

      return result;
    } catch (e) {
      throw EmotionError.badInput('ONNX inference failed: $e');
    }
  }

  /// Extract feature vector in the correct order for the model
  List<double> _extractFeatureVector(Map<String, double> features) {
    final featureVector = <double>[];

    // Map feature names (case-insensitive)
    final featureMap = <String, double>{};
    for (final entry in features.entries) {
      featureMap[entry.key.toLowerCase()] = entry.value;
    }

    for (final inputName in inputNames) {
      final key = inputName.toLowerCase();

      // Handle different naming conventions
      double? value;

      // Direct match first
      value = featureMap[key];

      // Fallback to common aliases
      if (value == null) {
        if (key == 'sdnn' || key == 'hrv_sdnn') {
          value = featureMap['sdnn'] ?? featureMap['hrv_sdnn'];
        } else if (key == 'rmssd') {
          value = featureMap['rmssd'];
        } else if (key == 'pnn50') {
          value = featureMap['pnn50'];
        } else if (key == 'mean_rr') {
          value = featureMap['mean_rr'];
        } else if (key == 'hr_mean' || key == 'hr') {
          value = featureMap['hr_mean'] ?? featureMap['hr'];
        } else if (key == 'hrv_hf') {
          value = featureMap['hrv_hf'] ?? featureMap['hf'];
        } else if (key == 'hrv_lf') {
          value = featureMap['hrv_lf'] ?? featureMap['lf'];
        } else if (key == 'hrv_hf_nu') {
          value = featureMap['hrv_hf_nu'] ?? featureMap['hf_nu'];
        } else if (key == 'hrv_lf_nu') {
          value = featureMap['hrv_lf_nu'] ?? featureMap['lf_nu'];
        } else if (key == 'hrv_lfhf') {
          value = featureMap['hrv_lfhf'] ?? featureMap['lfhf'];
        } else if (key == 'hrv_tp') {
          value = featureMap['hrv_tp'] ?? featureMap['tp'];
        } else if (key == 'hrv_sd1sd2') {
          value = featureMap['hrv_sd1sd2'] ?? featureMap['sd1sd2'];
        } else if (key == 'hrv_sampen') {
          value = featureMap['hrv_sampen'] ?? featureMap['sampen'];
        } else if (key == 'hrv_dfa_alpha1') {
          value = featureMap['hrv_dfa_alpha1'] ?? featureMap['dfa_alpha1'];
        } else if (key == 'mean_rr') {
          value = featureMap['mean_rr'] ?? featureMap['mean_rr'];
        }
      }

      if (value == null) {
        throw EmotionError.badInput('Missing required feature: $inputName');
      }

      featureVector.add(value);
    }

    // Verify we extracted exactly 14 features (no more, no less)
    if (inputNames.length != 14) {
      throw EmotionError.badInput(
        'Model configuration error: model expects ${inputNames.length} features, '
        'but this implementation requires exactly 14 features.',
      );
    }

    if (featureVector.length != 14) {
      throw EmotionError.badInput(
        'Feature count mismatch: expected exactly 14 features, '
        'extracted ${featureVector.length}. Model requires exactly 14 features: '
        '${inputNames.join(", ")}',
      );
    }

    return featureVector;
  }

  /// Get model metadata
  Map<String, dynamic> getMetadata() => {
    'id': modelId,
    'type': 'onnx',
    'labels': classNames,
    'feature_names': inputNames,
    'num_classes': classNames.length,
    'num_features': inputNames.length,
    'format': _metadata['format'],
    'created_utc': _metadata['created_utc'],
  };

  /// Validate model integrity
  bool validate() {
    try {
      if (inputNames.isEmpty) {
        return false;
      }
      if (classNames.isEmpty) {
        return false;
      }
      return true;
    } catch (e) {
      return false;
    }
  }

  /// Get feature names (for compatibility with LinearSvmModel interface)
  List<String> get featureNames => inputNames;

  /// Get labels (for compatibility with LinearSvmModel interface)
  List<String> get labels => classNames;

  /// Release resources
  Future<void> dispose() async {
    try {
      await _session.close();
    } catch (e) {
      // Ignore close errors
    }
  }
}
