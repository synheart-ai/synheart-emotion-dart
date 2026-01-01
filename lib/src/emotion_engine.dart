import 'dart:collection';

import 'emotion_config.dart';
import 'emotion_error.dart';
import 'emotion_result.dart';
import 'features.dart';

/// Data point for ring buffer
class _DataPoint {
  _DataPoint({
    required this.timestamp,
    required this.hr,
    required this.rrIntervalsMs,
    this.motion,
  });

  final DateTime timestamp;
  final double hr;
  final List<double> rrIntervalsMs;
  final Map<String, double>? motion;
}

/// Main emotion inference engine.
///
/// Processes biosignal data using a sliding window approach and produces
/// emotion predictions at configurable intervals.
class EmotionEngine {
  EmotionEngine._({required this.config, required this.model, this.onLog});

  /// Create engine from pretrained model
  factory EmotionEngine.fromPretrained(
    EmotionConfig config, {
    dynamic model,
    void Function(
      String level,
      String message, {
      Map<String, Object?>? context,
    })?
    onLog,
  }) {
    // Use provided model or default
    final inferenceModel = model;

    // Validate model has required interface
    if (inferenceModel != null) {
      if (!inferenceModel.toString().contains('Model')) {
        throw EmotionError.badInput('Invalid model type provided');
      }
    }

    return EmotionEngine._(config: config, model: inferenceModel, onLog: onLog);
  }

  /// Expected number of core HRV features.
  /// Note: For 14-feature models (ExtraTrees), this is 14.
  static const int expectedFeatureCount = 14;
  
  /// Check if model uses 14-feature extraction
  bool get _uses14Features =>
      config.modelId.contains('extratrees') ||
      config.modelId.contains('ExtraTrees') ||
      (model != null &&
          model.runtimeType.toString().contains('Onnx') &&
          (model as dynamic).inputNames.length == 14);

  /// Configuration for this emotion engine instance.
  final EmotionConfig config;

  /// The inference model used for emotion prediction.
  final dynamic model; // Can be LinearSvmModel or OnnxEmotionModel

  /// Ring buffer for sliding window
  final Queue<_DataPoint> _buffer = Queue<_DataPoint>();

  /// Last emission timestamp
  DateTime? _lastEmission;

  /// Logging callback
  void Function(String level, String message, {Map<String, Object?>? context})?
  onLog;

  /// Push new data point into the engine
  void push({
    required double hr,
    required List<double> rrIntervalsMs,
    required DateTime timestamp,
    Map<String, double>? motion,
  }) {
    try {
      // Validate input using physiological constants
      if (hr < FeatureExtractor.minValidHr ||
          hr > FeatureExtractor.maxValidHr) {
        _log(
          'warn',
          'Invalid HR value: $hr (valid range: '
              '${FeatureExtractor.minValidHr}-'
              '${FeatureExtractor.maxValidHr} BPM)',
        );
        return;
      }

      if (rrIntervalsMs.isEmpty) {
        _log('warn', 'Empty RR intervals');
        return;
      }

      // Add to ring buffer
      final dataPoint = _DataPoint(
        timestamp: timestamp,
        hr: hr,
        rrIntervalsMs: List.from(rrIntervalsMs),
        motion: motion,
      );

      _buffer.add(dataPoint);

      // Remove old data points outside window
      _trimBuffer();

      _log(
        'debug',
        'Pushed data point: HR=$hr, RR count=${rrIntervalsMs.length}',
      );
    } catch (e) {
      _log('error', 'Error pushing data point: $e');
    }
  }

  /// Consume ready results (throttled by step interval)
  ///
  /// Returns results synchronously (no await required), matching API
  /// specification across all platforms (Python, Kotlin, Swift).
  List<EmotionResult> consumeReady() {
    final results = <EmotionResult>[];

    if (model == null) {
      return results;
    }

    try {
      // Check if enough time has passed since last emission
      final now = DateTime.now().toUtc();
      if (_lastEmission != null &&
          now.difference(_lastEmission!).compareTo(config.step) < 0) {
        return results; // Not ready yet
      }

      // Check if we have enough data
      if (_buffer.length < 2) {
        return results; // Not enough data
      }

      // Extract features from current window
      final features = _extractWindowFeatures();
      if (features == null) {
        return results; // Feature extraction failed
      }

      // Run inference synchronously (Linear SVM model)
      // Note: ONNX async models are not supported in consumeReady()
      // to maintain API parity with Python, Kotlin, and Swift SDKs
      // which are all synchronous.
      final Map<String, double> probabilities;
      if (model.runtimeType.toString().contains('Onnx')) {
        // ONNX models require async - return empty for now
        // Callers should use consumeReadyAsync() for ONNX models
        return results;
      } else {
        // Linear SVM model is synchronous
        probabilities = model.predict(features);
      }

      // Create result
      final result = EmotionResult.fromInference(
        timestamp: now,
        probabilities: probabilities,
        features: features,
        model: model.getMetadata(),
      );

      results.add(result);
      _lastEmission = now;

      _log(
        'info',
        'Emitted result: ${result.emotion} '
            '(${(result.confidence * 100).toStringAsFixed(1)}%)',
      );
    } catch (e) {
      _log('error', 'Error during inference: $e');
    }

    return results;
  }

  /// Consume ready results asynchronously (for ONNX models)
  ///
  /// This method supports async ONNX model inference while maintaining
  /// the same throttling and windowing logic as consumeReady().
  Future<List<EmotionResult>> consumeReadyAsync() async {
    final results = <EmotionResult>[];

    if (model == null) {
      _log('warn', 'Model is null, cannot perform inference');
      return results;
    }

    try {
      // Check if enough time has passed since last emission
      final now = DateTime.now().toUtc();
      if (_lastEmission != null) {
        final timeSinceLastEmission = now.difference(_lastEmission!);
        if (timeSinceLastEmission.compareTo(config.step) < 0) {
          _log(
            'debug',
            'Step interval throttling: ${timeSinceLastEmission.inSeconds}s since last emission, need ${config.step.inSeconds}s',
          );
          return results; // Not ready yet
        }
      }

      // Check if we have enough data
      if (_buffer.length < 2) {
        _log('warn', 'Not enough data points in buffer: ${_buffer.length} < 2');
        return results; // Not enough data
      }

      // Extract features from current window
      Map<String, double>? features;
      try {
        features = _extractWindowFeatures();
        if (features == null) {
          _log('error', 'Feature extraction failed - returned null');
          return results; // Feature extraction failed
        }

        _log(
          'debug',
          'Features extracted successfully: ${features.keys.join(", ")} (${features.length} features)',
        );
      } catch (e, stackTrace) {
        _log('error', 'Feature extraction threw exception: $e');
        _log('error', 'Stack trace: $stackTrace');
        return results; // Feature extraction failed
      }

      // Run inference - support both sync and async models
      final Map<String, double> probabilities;
      if (model.runtimeType.toString().contains('Onnx')) {
        // ONNX model - use async prediction
        _log('debug', 'Running ONNX inference...');
        try {
          probabilities = await (model as dynamic).predictAsync(features);
          _log('debug', 'ONNX inference completed successfully');
        } catch (e, stackTrace) {
          _log('error', 'ONNX inference failed: $e');
          _log('error', 'Stack trace: $stackTrace');
          rethrow;
        }
      } else {
        // Linear SVM model is synchronous
        _log('debug', 'Running Linear SVM inference...');
        probabilities = model.predict(features);
        _log('debug', 'Linear SVM inference completed successfully');
      }

      // Create result
      final result = EmotionResult.fromInference(
        timestamp: now,
        probabilities: probabilities,
        features: features,
        model: model.getMetadata(),
      );

      results.add(result);
      _lastEmission = now;

      _log(
        'info',
        'Emitted result: ${result.emotion} '
            '(${(result.confidence * 100).toStringAsFixed(1)}%)',
      );
    } catch (e, stackTrace) {
      _log('error', 'Error during inference: $e');
      _log('error', 'Stack trace: $stackTrace');
    }

    return results;
  }

  /// Extract features from current window
  Map<String, double>? _extractWindowFeatures() {
    if (_buffer.isEmpty) {
      return null;
    }

    // Check if the oldest data point in buffer is at least window duration old
    // This ensures we have a full window of data from (now - window) to now
    // Allow 2 second tolerance for timing precision and data push intervals
    final now = DateTime.now().toUtc();
    final oldestDataAge = now.difference(_buffer.first.timestamp);
    final requiredAge = config.window - const Duration(seconds: 2); // 2 second tolerance
    if (oldestDataAge < requiredAge) {
      _log(
        'warn',
        'Buffer window insufficient: oldest data is ${oldestDataAge.inSeconds}s old, need ${config.window.inSeconds}s window. Need ${(requiredAge.inSeconds - oldestDataAge.inSeconds).toStringAsFixed(1)}s more.',
      );
      return null;
    }

    // Collect all HR values and RR intervals in window
    final hrValues = <double>[];
    final allRrIntervals = <double>[];
    Map<String, double>? motionAggregate;

    for (final point in _buffer) {
      hrValues.add(point.hr);
      allRrIntervals.addAll(point.rrIntervalsMs);

      // Aggregate motion data
      if (point.motion != null) {
        motionAggregate ??= <String, double>{};
        for (final entry in point.motion!.entries) {
          motionAggregate[entry.key] =
              (motionAggregate[entry.key] ?? 0.0) + entry.value;
        }
      }
    }

    // Check minimum RR count
    if (allRrIntervals.length < config.minRrCount) {
      _log(
        'warn',
        'Too few RR intervals: ${allRrIntervals.length} < ${config.minRrCount}',
      );
      return null;
    }

    // Extract features - use 14-feature extraction if model requires it
    final features = FeatureExtractor.extractFeatures(
      hrValues: hrValues,
      rrIntervalsMs: allRrIntervals,
      motion: motionAggregate,
      use14Features: _uses14Features,
    );

    // Apply personalization if configured
    if (config.hrBaseline != null) {
      // For 14-feature mode, HR is stored as 'HR', for legacy mode it's 'hr_mean'
      final hrKey = _uses14Features ? 'HR' : 'hr_mean';
      if (features.containsKey(hrKey)) {
        features[hrKey] = features[hrKey]! - config.hrBaseline!;
      }
    }

    return features;
  }

  /// Trim buffer to keep only data within window.
  ///
  /// Optimized implementation that removes all expired data points
  /// in a single pass to avoid repeated O(n) removeFirst() calls.
  void _trimBuffer() {
    if (_buffer.isEmpty) {
      return;
    }

    final cutoffTime = DateTime.now().toUtc().subtract(config.window);

    // Find index of first valid data point
    var firstValidIndex = 0;
    for (final point in _buffer) {
      if (!point.timestamp.isBefore(cutoffTime)) {
        break;
      }
      firstValidIndex++;
    }

    // Remove all expired data points at once if any found
    if (firstValidIndex > 0) {
      // Rebuild queue with only valid data points
      // (more efficient than repeated removeFirst)
      final validPoints = _buffer.skip(firstValidIndex).toList();
      _buffer
        ..clear()
        ..addAll(validPoints);
    }
  }

  /// Get current buffer statistics
  Map<String, dynamic> getBufferStats() {
    if (_buffer.isEmpty) {
      return {
        'count': 0,
        'duration_ms': 0,
        'hr_range': [0.0, 0.0],
        'rr_count': 0,
      };
    }

    final hrValues = _buffer.map((p) => p.hr).toList();
    final rrCount = _buffer.fold(0, (sum, p) => sum + p.rrIntervalsMs.length);
    final duration = _buffer.last.timestamp.difference(_buffer.first.timestamp);

    return {
      'count': _buffer.length,
      'duration_ms': duration.inMilliseconds,
      'hr_range': [
        hrValues.reduce((a, b) => a < b ? a : b),
        hrValues.reduce((a, b) => a > b ? a : b),
      ],
      'rr_count': rrCount,
    };
  }

  /// Clear all buffered data
  void clear() {
    _buffer.clear();
    _lastEmission = null;
    _log('info', 'Buffer cleared');
  }

  /// Log message with optional context
  void _log(String level, String message, {Map<String, Object?>? context}) {
    onLog?.call(level, message, context: context);
  }
}
