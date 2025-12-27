# Changelog

All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] - 2025-12-26

### Fixed
- Updated README dependency snippets and API reference to match the current package version and synchronous `consumeReady()` API

## [0.2.2] - 2025-12-07

### Changed
- Made `consumeReady()` method synchronous (removed async/await requirement) for API consistency across platforms (Python, Kotlin, Swift, Dart)
- Updated `expectedFeatureCount` from 5 to 3 to reflect actual model requirements (hr_mean, sdnn, rmssd)

### Fixed
- Fixed stream listener in `EmotionStream` to use synchronous `consumeReady()` call

## [0.2.1] - 2025-11-10

### Fixed
- Fixed README.md version number to match current package version (0.2.1)
- Fixed README.md API documentation: `consumeReady()` is now correctly documented as `Future<List<EmotionResult>>` (async method)
- Fixed README.md example code to properly use `await` with `consumeReady()` calls
- Fixed `EmotionError` to properly implement `Exception` interface, resolving `only_throw_errors` lint warnings
- Fixed directive ordering in library exports for better code organization
- Fixed Android example app `minSdkVersion` to 24 to match ONNX Runtime requirements
- Fixed CI/CD workflows to handle info-level linting issues gracefully
- Improved error handling in CI workflow's `all-checks-passed` job

### Added
- Added `.gitignore` file with comprehensive Flutter/Dart ignore patterns for build artifacts, IDE files, and OS-specific files

### Changed
- Updated CI workflows to use `--no-fatal-infos` flag for static analysis, allowing info-level suggestions without failing builds
- Enhanced CI error reporting with better diagnostics for failed jobs

## [0.2.0] - 2025-11-07

### Breaking Changes
- **Removed `LinearSvmModel` class**: The `LinearSvmModel` class and `model_linear_svm.dart` file have been removed. The package now focuses exclusively on ONNX models.
- **Removed `json_linear_model.dart` export**: The dead export for `json_linear_model.dart` has been removed (file was missing).
- **Package structure change**: Moved `lib/` directory from repository root to `sdks/flutter/lib/` to follow proper Flutter package conventions.

### Migration Guide

If you were using `LinearSvmModel`, migrate to `OnnxEmotionModel`:

```dart
// Old (0.1.0)
import 'package:synheart_emotion/synheart_emotion.dart';

final model = LinearSvmModel.fromArrays(
  modelId: 'wesad_emotion_v1_0',
  version: '1.0',
  labels: ['Amused', 'Calm', 'Stressed'],
  featureNames: ['hr_mean', 'sdnn', 'rmssd'],
  weights: [...],
  biases: [...],
  mu: {...},
  sigma: {...},
);

final engine = EmotionEngine.fromPretrained(
  config,
  model: model,
);

// New (0.2.0)
import 'package:synheart_emotion/synheart_emotion.dart';

final model = await OnnxEmotionModel.loadFromAsset(
  modelAssetPath: 'assets/ml/extratrees_wrist_all_v1_0.onnx',
  metaAssetPath: 'assets/ml/extratrees_wrist_all_v1_0.meta.json',
);

final engine = EmotionEngine.fromPretrained(
  config,
  model: model,
);
```

### Fixed
- Fixed Flutter package structure: `lib/` directory is now properly located in `sdks/flutter/lib/` where `pubspec.yaml` is located, enabling proper pub.dev publishing.

### Changed
- Package now exclusively uses ONNX models for emotion inference.
- Improved package structure alignment with Flutter package conventions.

## [0.1.0] - 2025-01-30

### Added
- Initial release of synheart_emotion Flutter package
- `EmotionEngine` for processing biosignal data
- `OnnxEmotionModel` for ONNX model loading and inference
- `FeatureExtractor` for HRV feature extraction
- Support for three emotion classes: Amused, Calm, Stressed
- Real-time emotion streaming with `EmotionStream`
- Comprehensive test suite

