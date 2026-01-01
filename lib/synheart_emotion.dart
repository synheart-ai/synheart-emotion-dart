/// On-device emotion inference from biosignals (heart rate and RR intervals).
///
/// This library provides real-time emotion detection from heart rate and
/// RR interval data, designed for wearable devices and health monitoring apps.
///
/// ## Features
///
/// * **Privacy-first**: All processing happens on-device
/// * **Real-time**: <10ms inference latency
/// * **Two emotion states**: Baseline, Stress
/// * **Sliding window**: 120s window with 60s step (default, configurable)
/// * **14 HRV features**: Comprehensive feature extraction
/// * **Stream API**: Reactive programming support
///
/// ## Quick Start
///
/// ```dart
/// import 'package:synheart_emotion/synheart_emotion.dart';
///
/// // Create engine with default configuration
/// final config = EmotionConfig();
/// final engine = EmotionEngine.fromPretrained(config);
///
/// // Push data from wearable
/// engine.push(
///   hr: 72.0,
///   rrIntervalsMs: [850.0, 820.0, 830.0, ...],
///   timestamp: DateTime.now(),
/// );
///
/// // Get inference result when ready (async for ONNX models)
/// final results = await engine.consumeReadyAsync();
/// for (final result in results) {
///   print('Emotion: ${result.emotion}');
///   print('Confidence: ${result.confidence}');
/// }
/// ```
///
/// ## Integration with SWIP SDK
///
/// This library is designed to work with the SWIP SDK for impact measurement.
/// See `example/lib/integration_example.dart` for complete integration examples.
///
/// ## Privacy & Security
///
/// **IMPORTANT**: This library uses demo placeholder model weights that are
/// NOT trained on real biosignal data. For production use, you must provide
/// your own trained model weights.
///
/// All processing happens on-device. No data is sent to external servers.
///
/// See documentation for details.
library synheart_emotion;

// Core engine and models
export 'src/emotion_config.dart';
export 'src/emotion_engine.dart';
export 'src/emotion_error.dart';
export 'src/emotion_result.dart';
export 'src/emotion_stream.dart';
export 'src/features.dart';
export 'src/hrv_features_complete.dart';
export 'src/onnx_model.dart';
