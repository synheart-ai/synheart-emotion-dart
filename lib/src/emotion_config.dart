import 'package:meta/meta.dart';

/// Configuration for the emotion inference engine
@immutable
class EmotionConfig {
  const EmotionConfig({
    this.modelId = 'extratrees_w120s60_binary_v1_0',
    this.window = const Duration(seconds: 120),
    this.step = const Duration(seconds: 60),
    this.minRrCount = 30,
    this.returnAllProbas = true,
    this.hrBaseline,
    this.priors,
  });

  /// Model identifier (default: extratrees_w120s60_binary_v1_0)
  /// Supports: ExtraTrees_60_5, ExtraTrees_120_5, ExtraTrees_120_60
  final String modelId;

  /// Rolling window size for feature calculation (default: 120s)
  final Duration window;

  /// Emission cadence for results (default: 60s)
  final Duration step;

  /// Minimum RR intervals required for inference (default: 30)
  final int minRrCount;

  /// Whether to return all label probabilities (default: true)
  final bool returnAllProbas;

  /// Optional HR baseline for personalization
  final double? hrBaseline;

  /// Optional label priors for calibration
  final Map<String, double>? priors;

  /// Create a copy with modified fields
  EmotionConfig copyWith({
    String? modelId,
    Duration? window,
    Duration? step,
    int? minRrCount,
    bool? returnAllProbas,
    double? hrBaseline,
    Map<String, double>? priors,
  }) => EmotionConfig(
    modelId: modelId ?? this.modelId,
    window: window ?? this.window,
    step: step ?? this.step,
    minRrCount: minRrCount ?? this.minRrCount,
    returnAllProbas: returnAllProbas ?? this.returnAllProbas,
    hrBaseline: hrBaseline ?? this.hrBaseline,
    priors: priors ?? this.priors,
  );

  @override
  String toString() =>
      'EmotionConfig(modelId: $modelId, '
      'window: ${window.inSeconds}s, '
      'step: ${step.inSeconds}s, minRrCount: $minRrCount)';

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) {
      return true;
    }
    return other is EmotionConfig &&
        other.modelId == modelId &&
        other.window == window &&
        other.step == step &&
        other.minRrCount == minRrCount &&
        other.returnAllProbas == returnAllProbas &&
        other.hrBaseline == hrBaseline &&
        _mapEquals(other.priors, priors);
  }

  @override
  int get hashCode => Object.hash(
    modelId,
    window,
    step,
    minRrCount,
    returnAllProbas,
    hrBaseline,
    priors,
  );

  bool _mapEquals(Map<String, double>? a, Map<String, double>? b) {
    if (a == null && b == null) {
      return true;
    }
    if (a == null || b == null) {
      return false;
    }
    if (a.length != b.length) {
      return false;
    }
    for (final key in a.keys) {
      if (a[key] != b[key]) {
        return false;
      }
    }
    return true;
  }
}
