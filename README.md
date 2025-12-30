# Synheart Emotion

**On-device emotion inference from biosignals (HR/RR) for Flutter applications**

[![pub package](https://img.shields.io/pub/v/synheart_emotion.svg)](https://pub.dev/packages/synheart_emotion)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/synheart-ai/synheart-emotion-flutter/actions/workflows/ci.yml/badge.svg)](https://github.com/synheart-ai/synheart-emotion-flutter/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/synheart-ai/synheart-emotion-flutter/branch/main/graph/badge.svg)](https://codecov.io/gh/synheart-ai/synheart-emotion-flutter)
[![Platform](https://img.shields.io/badge/Platform-iOS%20%7C%20Android-lightgrey.svg)](https://flutter.dev)
[![Dart](https://img.shields.io/badge/Dart-3.7.0+-blue.svg)](https://dart.dev)
[![Flutter](https://img.shields.io/badge/Flutter-3.24.0+-blue.svg)](https://flutter.dev)

## 🚀 Features

- **📱 Cross-Platform**: Works on iOS and Android
- **🔄 Real-Time Inference**: Live emotion detection from heart rate and RR intervals
- **🧠 On-Device Processing**: All computations happen locally for privacy
- **📊 Unified Output**: Consistent emotion labels with confidence scores
- **🔒 Privacy-First**: No raw biometric data leaves your device
- **⚡ High Performance**: < 1s inference latency 
- **🧬 14 HRV Features**: Comprehensive feature extraction (time-domain, frequency-domain, non-linear)
- **🤖 ONNX Models**: ExtraTrees classifiers optimized for on-device inference

## 📦 Installation

Add `synheart_emotion` to your `pubspec.yaml`:

```yaml
dependencies:
  synheart_emotion: ^0.2.3
```

Then run:

```bash
flutter pub get
```

## 🎯 Quick Start

### Basic Usage

```dart
import 'package:synheart_emotion/synheart_emotion.dart';

void main() async {
  // Initialize the emotion engine (default: 120s window, 60s step)
  final engine = EmotionEngine.fromPretrained(
    const EmotionConfig(),
  );

  // Push biometric data
  engine.push(
    hr: 72.0,
    rrIntervalsMs: [823, 810, 798, 815, 820],
    timestamp: DateTime.now().toUtc(),
  );

  // Get emotion results (async for ONNX models)
  final results = await engine.consumeReadyAsync();
  for (final result in results) {
    print('Emotion: ${result.emotion} (${(result.confidence * 100).toStringAsFixed(1)}%)');
  }
}
```

### Real-Time Streaming

```dart
// Stream emotion results
final emotionStream = EmotionStream.emotionStream(
  engine,
  tickStream, // Your biometric data stream
);

await for (final result in emotionStream) {
  print('Current emotion: ${result.emotion}');
  print('Probabilities: ${result.probabilities}');
}
```

### Integration with synheart-wear

**synheart_emotion** works independently but integrates seamlessly with [synheart-wear](https://github.com/synheart-ai/synheart-wear) for real wearable data.



First, add both to your `pubspec.yaml`:

```yaml
dependencies:
  synheart_wear: ^0.1.0    # For wearable data
  synheart_emotion: ^0.2.3  # For emotion inference
```

Then integrate in your app:

```dart
import 'package:synheart_wear/synheart_wear.dart';
import 'package:synheart_emotion/synheart_emotion.dart';

// Initialize both SDKs
final wear = SynheartWear();
final emotionEngine = EmotionEngine.fromPretrained(
  const EmotionConfig(), // Default: 120s window, 60s step
);

await wear.initialize();

// Stream wearable data to emotion engine
wear.streamHR(interval: Duration(seconds: 1)).listen((metrics) {
  emotionEngine.push(
    hr: metrics.getMetric(MetricType.hr),
    rrIntervalsMs: metrics.getMetric(MetricType.rrIntervals),
    timestamp: DateTime.now().toUtc(),
  );
  
  // Get emotion results (async for ONNX models)
  final emotions = await emotionEngine.consumeReadyAsync();
  for (final emotion in emotions) {
    // Use emotion data in your app
    updateUI(emotion);
  }
});
```

See `examples/lib/integration_example.dart` for complete integration examples.

## 📊 Supported Emotions

The library currently supports two emotion categories:

- **😌 Baseline**: Relaxed, peaceful emotional state  
- **😰 Stress**: Anxious, tense emotional state

## 🧠 Model Architecture

The library uses **ExtraTrees (Extremely Randomized Trees)** classifiers trained on the WESAD dataset:

- **14 HRV Features**: Time-domain, frequency-domain, and non-linear metrics
- **Binary Classification**: Baseline vs Stress detection
- **ONNX Format**: Optimized for on-device inference
- **Accuracy**: 78.4% on WESAD validation set (LOSO CV)
- **F1 Score**: 72.6% on WESAD validation set (LOSO CV)

### Available Models

- `ExtraTrees_120_60`: 120-second window, 60-second step (default)
- `ExtraTrees_60_5`: 60-second window, 5-second step
- `ExtraTrees_120_5`: 120-second window, 5-second step

### Feature Extraction

The library extracts 14 HRV features in the following order:

**Time-domain features:**
- RMSSD (Root Mean Square of Successive Differences)
- Mean_RR (Mean RR interval)
- HRV_SDNN (Standard Deviation of NN intervals)
- pNN50 (Percentage of successive differences > 50ms)

**Frequency-domain features:**
- HRV_HF (High Frequency power)
- HRV_LF (Low Frequency power)
- HRV_HF_nu (Normalized HF)
- HRV_LF_nu (Normalized LF)
- HRV_LFHF (LF/HF ratio)
- HRV_TP (Total Power)

**Non-linear features:**
- HRV_SD1SD2 (Poincaré plot ratio)
- HRV_Sampen (Sample Entropy)
- HRV_DFA_alpha1 (Detrended Fluctuation Analysis)

**Heart Rate:**
- HR (Heart Rate in BPM)

## 🔧 API Reference

### EmotionEngine

The main class for emotion inference:

```dart
class EmotionEngine {
  // Create engine with pretrained model
  factory EmotionEngine.fromPretrained(
    EmotionConfig config, {
    dynamic model,
    void Function(String level, String message, {Map<String, Object?>? context})? onLog,
  });

  // Push new biometric data
  void push({
    required double hr,
    required List<double> rrIntervalsMs,
    required DateTime timestamp,
    Map<String, double>? motion,
  });

  // Get ready emotion results (async - for ONNX models)
  Future<List<EmotionResult>> consumeReadyAsync();

  // Get buffer statistics
  Map<String, dynamic> getBufferStats();

  // Clear all buffered data
  void clear();
}
```

### EmotionConfig

Configuration for the emotion engine:

```dart
class EmotionConfig {
  final String modelId;                 // Model identifier (default: extratrees_w120s60_binary_v1_0)
  final Duration window;                // Rolling window size (default: 120s)
  final Duration step;                  // Emission cadence (default: 60s)
  final int minRrCount;                 // Min RR intervals needed (default: 30)
  final bool returnAllProbas;           // Return all probabilities (default: true)
  final double? hrBaseline;             // Optional HR personalization
  final Map<String,double>? priors;     // Optional label priors
}
```

### EmotionResult

Result of emotion inference:

```dart
class EmotionResult {
  final DateTime timestamp;             // When inference was performed
  final String emotion;                 // Predicted emotion (top-1)
  final double confidence;              // Confidence score (0.0-1.0)
  final Map<String, double> probabilities; // All label probabilities
  final Map<String, double> features;   // Extracted features
  final Map<String, dynamic> model;     // Model metadata
}
```

## 🔒 Privacy & Security

- **On-Device Processing**: All emotion inference happens locally
- **No Data Retention**: Raw biometric data is not retained after processing
- **No Network Calls**: No data is sent to external servers
- **Privacy-First Design**: No built-in storage - you control what gets persisted
- **Real Trained Models**: Uses WESAD-trained ExtraTrees models with 78.4% accuracy (72.6% F1 score)
- **14-Feature Extraction**: Comprehensive HRV analysis including time-domain, frequency-domain, and non-linear metrics

## 📱 Example App

Check out the complete examples in the [synheart-emotion repository](https://github.com/synheart-ai/synheart-emotion/tree/main/examples):

```bash
# Clone the main repository for examples
git clone https://github.com/synheart-ai/synheart-emotion.git
cd synheart-emotion/examples
flutter pub get
flutter run
```

The example demonstrates:
- Real-time emotion detection
- Probability visualization
- Buffer management
- Logging system

## 🧪 Testing

Run the test suite:

```bash
flutter test
```

Run benchmarks:

```bash
flutter test test/benchmarks_test.dart
```

Tests cover:
- Feature extraction accuracy
- Model inference performance
- Edge case handling
- Memory usage patterns
- Integration with synheart-core EmotionHead
- HSI schema compatibility validation
- Time-series data processing with ring buffer
- Push/consumeReadyAsync API pattern

## 📊 Performance

**Target Performance (mid-range phone):**
- **Latency**: < 10ms per inference (ONNX models)
- **Model Size**: ~200-300 KB per model
- **CPU Usage**: < 3% during active streaming
- **Memory**: < 5 MB (engine + buffers + ONNX runtime)
- **Accuracy**: 78.4% on WESAD dataset (binary classification: Baseline vs Stress)
- **F1 Score**: 72.6% on WESAD dataset (LOSO CV)

**Benchmarks:**
- 14-feature extraction: < 3ms
- ONNX model inference: < 5ms
- Full pipeline: < 10ms

## 🏗️ Architecture

```
Biometric Data (HR, RR)
         │
         ▼
┌─────────────────────┐
│   EmotionEngine     │
│  [RingBuffer]       │
│  [14-Feature        │
│   Extractor]        │
│  [ONNX Inference]   │
└─────────────────────┘
         │
         ▼
   EmotionResult
   (Baseline/Stress)
         │
         ▼
    Your App

```

### Feature Extraction Pipeline

1. **Time-domain features**: RMSSD, Mean_RR, SDNN, pNN50
2. **Frequency-domain features**: HF, LF, HF_nu, LF_nu, LFHF, TP (via FFT)
3. **Non-linear features**: SD1SD2, Sample Entropy, DFA alpha1
4. **Heart Rate**: Calculated from Mean_RR
```

## 🔗 Integration

### With synheart-core (HSI)

**synheart_emotion** is designed to integrate seamlessly with [synheart-core](https://github.com/synheart-ai/synheart-core) as part of the Human State Interface (HSI) system:

```dart
import 'package:synheart_core/synheart_core.dart';
import 'package:synheart_emotion/synheart_emotion.dart';

// Initialize synheart-core (includes emotion capability)
await Synheart.initialize(
  userId: 'user_123',
  config: SynheartConfig(
    enableWear: true,
    enableBehavior: true,
  ),
);

// Enable emotion interpretation layer (powered by synheart-emotion)
await Synheart.enableEmotion();

// Get emotion updates through HSI
Synheart.onEmotionUpdate.listen((emotion) {
  print('Baseline: ${emotion.baseline}');
  print('Stress: ${emotion.stress}');
});
```

**HSI Schema Compatibility:**
- EmotionResult from synheart-emotion maps to HSI EmotionState
- Output validated against HSI_SPECIFICATION.md
- Comprehensive integration tests ensure compatibility

See the [synheart-core documentation](https://github.com/synheart-ai/synheart-core) for more details on HSI integration.

### With synheart-wear

Perfect integration with the Synheart Wear SDK for real wearable data:

```dart
// Stream from Apple Watch, Fitbit, etc.
final wearStream = synheartWear.streamHR();
final emotionStream = EmotionStream.emotionStream(engine, wearStream);
```

### With swip-core

Feed emotion results into the SWIP impact measurement system:

```dart
for (final emotion in emotionResults) {
  swipCore.ingestEmotion(emotion);
}
```

## 📄 License

Apache 2.0 License

## 🤝 Contributing

We welcome contributions! See our [Contributing Guidelines](https://github.com/synheart-ai/synheart-emotion/blob/main/CONTRIBUTING.md) for details.

## 🔗 Links

- **Main Repository**: [synheart-emotion](https://github.com/synheart-ai/synheart-emotion) (Source of Truth)
- **Documentation**: [RFC E1.1](https://github.com/synheart-ai/synheart-emotion/blob/main/docs/RFC-E1.1.md)
- **Model Card**: [Model Card](https://github.com/synheart-ai/synheart-emotion/blob/main/docs/MODEL_CARD.md)
- **Examples**: [Examples](https://github.com/synheart-ai/synheart-emotion/tree/main/examples)
- **Models**: [Pre-trained Models](https://github.com/synheart-ai/synheart-emotion/tree/main/models)
- **Tools**: [Development Tools](https://github.com/synheart-ai/synheart-emotion/tree/main/tools)
- **Synheart Wear**: [synheart-wear](https://github.com/synheart-ai/synheart-wear)
- **Synheart AI**: [synheart.ai](https://synheart.ai)
- **Issues**: [GitHub Issues](https://github.com/synheart-ai/synheart-emotion-flutter/issues)

## 👥 Authors

- **Synheart AI Team** - _Initial work_, _RFC Design & Architecture_

---

**Made with ❤️ by the Synheart AI Team**

_Technology with a heartbeat._

## Patent Pending Notice

This project is provided under an open-source license. Certain underlying systems, methods, and architectures described or implemented herein may be covered by one or more pending patent applications.

Nothing in this repository grants any license, express or implied, to any patents or patent applications, except as provided by the applicable open-source license.
