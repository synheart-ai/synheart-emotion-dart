import 'dart:math';
import 'hrv_features_complete.dart';

/// Feature extraction utilities for emotion inference.
///
/// Provides methods for extracting heart rate variability (HRV) metrics
/// from biosignal data. Supports both legacy 5-feature extraction and
/// new 14-feature extraction for ExtraTrees models.
class FeatureExtractor {
  /// Minimum valid RR interval in milliseconds (300ms = 200 BPM).
  static const double minValidRrMs = 300;

  /// Maximum valid RR interval in milliseconds (2000ms = 30 BPM).
  static const double maxValidRrMs = 2000;

  /// Maximum allowed jump between successive RR intervals in milliseconds.
  ///
  /// This threshold helps detect and remove artifacts from RR interval data.
  /// A jump > 250ms between consecutive intervals likely indicates noise.
  static const double maxRrJumpMs = 250;

  /// Minimum heart rate value considered valid (in BPM).
  static const double minValidHr = 30;

  /// Maximum heart rate value considered valid (in BPM).
  static const double maxValidHr = 300;

  /// Extract HR mean from a list of HR values.
  ///
  /// Returns 0.0 if the input list is empty.
  static double extractHrMean(List<double> hrValues) {
    if (hrValues.isEmpty) {
      return 0;
    }
    return hrValues.reduce((a, b) => a + b) / hrValues.length;
  }

  /// Extract SDNN (standard deviation of NN intervals) from RR intervals
  static double extractSdnn(List<double> rrIntervalsMs) {
    if (rrIntervalsMs.length < 2) {
      return 0;
    }

    // Clean RR intervals (remove outliers)
    final cleaned = _cleanRrIntervals(rrIntervalsMs);
    if (cleaned.length < 2) {
      return 0;
    }

    // Calculate standard deviation (sample std, N-1 denominator)
    final mean = cleaned.reduce((a, b) => a + b) / cleaned.length;
    final variance =
        cleaned.map((x) => pow(x - mean, 2)).reduce((a, b) => a + b) /
        (cleaned.length - 1);
    return sqrt(variance);
  }

  /// Extract RMSSD (root mean square of successive differences)
  /// from RR intervals
  static double extractRmssd(List<double> rrIntervalsMs) {
    if (rrIntervalsMs.length < 2) {
      return 0;
    }

    // Clean RR intervals
    final cleaned = _cleanRrIntervals(rrIntervalsMs);
    if (cleaned.length < 2) {
      return 0;
    }

    // Calculate successive differences
    var sumSquaredDiffs = 0.0;
    for (var i = 1; i < cleaned.length; i++) {
      final diff = cleaned[i] - cleaned[i - 1];
      sumSquaredDiffs += diff * diff;
    }

    // Root mean square
    return sqrt(sumSquaredDiffs / (cleaned.length - 1));
  }

  /// Extract pNN50 (percentage of successive RR intervals
  /// differing by more than 50ms)
  static double extractPnn50(List<double> rrIntervalsMs) {
    if (rrIntervalsMs.length < 2) {
      return 0;
    }

    // Clean RR intervals
    final cleaned = _cleanRrIntervals(rrIntervalsMs);
    if (cleaned.length < 2) {
      return 0;
    }

    // Count successive differences > 50ms
    var count = 0;
    for (var i = 1; i < cleaned.length; i++) {
      if ((cleaned[i] - cleaned[i - 1]).abs() > 50) {
        count++;
      }
    }

    // Return percentage
    return (count / (cleaned.length - 1)) * 100;
  }

  /// Extract Mean RR interval from RR intervals
  static double extractMeanRr(List<double> rrIntervalsMs) {
    if (rrIntervalsMs.isEmpty) {
      return 0;
    }

    // Clean RR intervals
    final cleaned = _cleanRrIntervals(rrIntervalsMs);
    if (cleaned.isEmpty) {
      return 0;
    }

    return cleaned.reduce((a, b) => a + b) / cleaned.length;
  }

  /// Extract all features for emotion inference
  ///
  /// Supports both legacy 5-feature mode and new 14-feature mode.
  /// Set [use14Features] to true to use the complete 14-feature extraction
  /// required for ExtraTrees_120_60 and similar models.
  static Map<String, double> extractFeatures({
    required List<double> hrValues,
    required List<double> rrIntervalsMs,
    Map<String, double>? motion,
    bool use14Features = false,
  }) {
    if (use14Features) {
      // Use 14-feature extraction for ExtraTrees models
      // Compute mean HR from provided HR values if available, otherwise derive from RR
      double? meanHr;
      if (hrValues.isNotEmpty) {
        // Use mean of provided HR values
        meanHr = hrValues.reduce((a, b) => a + b) / hrValues.length;
      }

      final featureList = HrvFeaturesComplete.extractAllFeatures(
        rrIntervalsMs,
        meanHr: meanHr,
      );

      // Map to feature names expected by model (must match metadata input_names exactly)
      // Order: ['RMSSD', 'Mean_RR', 'HRV_SDNN', 'pNN50',
      // 'HRV_HF', 'HRV_LF', 'HRV_HF_nu', 'HRV_LF_nu', 'HRV_LFHF', 'HRV_TP',
      // 'HRV_SD1SD2', 'HRV_Sampen', 'HRV_DFA_alpha1', 'HR']
      // These names must exactly match the input_names in the ONNX model metadata
      final featureNames = [
        'RMSSD', // 0: Root Mean Square of Successive Differences
        'Mean_RR', // 1: Mean RR interval
        'HRV_SDNN', // 2: Standard Deviation of NN intervals
        'pNN50', // 3: Percentage of successive differences > 50ms
        'HRV_HF', // 4: High Frequency power
        'HRV_LF', // 5: Low Frequency power
        'HRV_HF_nu', // 6: Normalized HF
        'HRV_LF_nu', // 7: Normalized LF
        'HRV_LFHF', // 8: LF/HF ratio
        'HRV_TP', // 9: Total Power
        'HRV_SD1SD2', // 10: Poincaré plot ratio
        'HRV_Sampen', // 11: Sample Entropy
        'HRV_DFA_alpha1', // 12: Detrended Fluctuation Analysis
        'HR', // 13: Heart Rate in BPM
      ];

      // Verify we have exactly 14 features
      if (featureList.length != 14) {
        throw ArgumentError(
          'Expected 14 features, got ${featureList.length}. '
          'Feature extraction may have failed.',
        );
      }

      final features = <String, double>{};
      for (var i = 0; i < featureList.length && i < featureNames.length; i++) {
        features[featureNames[i]] = featureList[i];
      }

      // Add motion features if provided
      if (motion != null) {
        features.addAll(motion);
      }

      return features;
    } else {
      // Legacy 5-feature extraction
      final features = <String, double>{
        'hr_mean': extractHrMean(hrValues),
        'sdnn': extractSdnn(rrIntervalsMs),
        'rmssd': extractRmssd(rrIntervalsMs),
        'pnn50': extractPnn50(rrIntervalsMs),
        'mean_rr': extractMeanRr(rrIntervalsMs),
      };

      // Add motion features if provided
      if (motion != null) {
        features.addAll(motion);
      }

      return features;
    }
  }

  /// Extract 14 features as a list in the exact order expected by ExtraTrees
  /// models.
  ///
  /// Returns features in order: [RMSSD, Mean_RR, HRV_SDNN, pNN50, HRV_HF,
  /// HRV_LF, HRV_HF_nu, HRV_LF_nu, HRV_LFHF, HRV_TP, HRV_SD1SD2, HRV_Sampen,
  /// HRV_DFA_alpha1, HR]
  ///
  /// [meanHr] is optional - if provided, will be used as the HR feature instead
  /// of computing it from mean RR interval.
  static List<double> extract14Features(
    List<double> rrIntervalsMs, {
    double? meanHr,
  }) {
    return HrvFeaturesComplete.extractAllFeatures(
      rrIntervalsMs,
      meanHr: meanHr,
    );
  }

  /// Clean RR intervals by removing physiologically invalid values
  /// and artifacts.
  ///
  /// Removes:
  /// - RR intervals outside valid range
  ///   ([minValidRrMs] to [maxValidRrMs])
  /// - Large jumps between successive intervals (> [maxRrJumpMs])
  ///
  /// Returns filtered list of clean RR intervals.
  static List<double> _cleanRrIntervals(List<double> rrIntervalsMs) {
    if (rrIntervalsMs.isEmpty) {
      return [];
    }

    final cleaned = <double>[];
    double? prevValue;

    for (final rr in rrIntervalsMs) {
      // Skip outliers outside physiological range
      if (rr < minValidRrMs || rr > maxValidRrMs) {
        continue;
      }

      // Skip large jumps that likely indicate artifacts
      if (prevValue != null && (rr - prevValue).abs() > maxRrJumpMs) {
        continue;
      }

      cleaned.add(rr);
      prevValue = rr;
    }

    return cleaned;
  }

  /// Validate feature vector for model compatibility
  static bool validateFeatures(
    Map<String, double> features,
    List<String> requiredFeatures,
  ) {
    for (final feature in requiredFeatures) {
      if (!features.containsKey(feature)) {
        return false;
      }
      if (features[feature]!.isNaN || features[feature]!.isInfinite) {
        return false;
      }
    }
    return true;
  }

  /// Normalize features using training statistics
  static Map<String, double> normalizeFeatures(
    Map<String, double> features,
    Map<String, double> mu,
    Map<String, double> sigma,
  ) {
    final normalized = <String, double>{};

    for (final entry in features.entries) {
      final featureName = entry.key;
      final value = entry.value;

      if (mu.containsKey(featureName) && sigma.containsKey(featureName)) {
        final mean = mu[featureName]!;
        final std = sigma[featureName]!;

        // Avoid division by zero
        if (std > 0) {
          normalized[featureName] = (value - mean) / std;
        } else {
          normalized[featureName] = 0;
        }
      } else {
        // Keep original value if no normalization params
        normalized[featureName] = value;
      }
    }

    return normalized;
  }
}
