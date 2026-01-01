import 'dart:math';

/// Complete HRV Feature Extraction for WESAD Model
///
/// Extracts all 14 features required for emotion inference:
/// Time-domain, frequency-domain, and non-linear metrics
class HrvFeaturesComplete {
  // ============================================================================
  // TIME DOMAIN FEATURES (Already implemented)
  // ============================================================================

  /// Compute RMSSD (Root Mean Square of Successive Differences)
  static double computeRmssd(List<double> rrIntervalsMs) {
    if (rrIntervalsMs.length < 2) return 0.0;

    final validRr =
        rrIntervalsMs.where((rr) => rr >= 300 && rr <= 2000).toList();
    if (validRr.length < 2) return 0.0;

    double sumSquaredDiffs = 0.0;
    for (int i = 1; i < validRr.length; i++) {
      final diff = validRr[i] - validRr[i - 1];
      sumSquaredDiffs += diff * diff;
    }

    return sqrt(sumSquaredDiffs / (validRr.length - 1));
  }

  /// Compute Mean RR interval
  static double computeMeanRr(List<double> rrIntervalsMs) {
    if (rrIntervalsMs.isEmpty) return 0.0;

    final validRr =
        rrIntervalsMs.where((rr) => rr >= 300 && rr <= 2000).toList();
    if (validRr.isEmpty) return 0.0;

    return validRr.reduce((a, b) => a + b) / validRr.length;
  }

  /// Compute SDNN (Standard Deviation of NN intervals)
  static double computeSdnn(List<double> rrIntervalsMs) {
    if (rrIntervalsMs.length < 2) return 0.0;

    final validRr =
        rrIntervalsMs.where((rr) => rr >= 300 && rr <= 2000).toList();
    if (validRr.length < 2) return 0.0;

    final mean = validRr.reduce((a, b) => a + b) / validRr.length;
    final variance =
        validRr.map((x) => pow(x - mean, 2)).reduce((a, b) => a + b) /
        (validRr.length - 1);

    return sqrt(variance);
  }

  /// Compute pNN50 (Percentage of successive differences > 50ms)
  static double computePnn50(List<double> rrIntervalsMs) {
    if (rrIntervalsMs.length < 2) return 0.0;

    final validRr =
        rrIntervalsMs.where((rr) => rr >= 300 && rr <= 2000).toList();
    if (validRr.length < 2) return 0.0;

    int count = 0;
    for (int i = 1; i < validRr.length; i++) {
      if ((validRr[i] - validRr[i - 1]).abs() > 50) {
        count++;
      }
    }

    return (count / (validRr.length - 1)) * 100;
  }

  // ============================================================================
  // FREQUENCY DOMAIN FEATURES (FFT-based)
  // ============================================================================

  /// Compute frequency domain features using FFT
  /// Returns: {HF, LF, VLF, HF_nu, LF_nu, LFHF, TP}
  static Map<String, double> computeFrequencyDomain(
    List<double> rrIntervalsMs,
  ) {
    if (rrIntervalsMs.length < 10) {
      return {
        'HF': 0.0,
        'LF': 0.0,
        'VLF': 0.0,
        'HF_nu': 0.0,
        'LF_nu': 0.0,
        'LFHF': 0.0,
        'TP': 0.0,
      };
    }

    // Resample RR intervals to uniform time series (4 Hz recommended)
    final resampledRr = _resampleRrIntervals(rrIntervalsMs, samplingRate: 4.0);

    if (resampledRr.length < 16) {
      return {
        'HF': 0.0,
        'LF': 0.0,
        'VLF': 0.0,
        'HF_nu': 0.0,
        'LF_nu': 0.0,
        'LFHF': 0.0,
        'TP': 0.0,
      };
    }

    // Apply Welch's method for PSD estimation
    final psd = _welchPsd(resampledRr, samplingRate: 4.0);

    // Define frequency bands (in Hz)
    const vlfBand = [0.0033, 0.04]; // VLF: 0.0033-0.04 Hz
    const lfBand = [0.04, 0.15]; // LF: 0.04-0.15 Hz
    const hfBand = [0.15, 0.4]; // HF: 0.15-0.4 Hz

    // Calculate power in each band
    final vlf = _bandPower(psd, vlfBand[0], vlfBand[1], 4.0);
    final lf = _bandPower(psd, lfBand[0], lfBand[1], 4.0);
    final hf = _bandPower(psd, hfBand[0], hfBand[1], 4.0);
    final tp = vlf + lf + hf;

    // Normalized units (exclude VLF)
    final lfhfSum = lf + hf;
    final lfNu = lfhfSum > 0 ? (lf / lfhfSum) : 0.0;
    final hfNu = lfhfSum > 0 ? (hf / lfhfSum) : 0.0;

    // LF/HF ratio
    final lfhf = hf > 0 ? (lf / hf) : 0.0;

    return {
      'HF': hf,
      'LF': lf,
      'VLF': vlf,
      'HF_nu': hfNu,
      'LF_nu': lfNu,
      'LFHF': lfhf,
      'TP': tp,
    };
  }

  /// Resample RR intervals to uniform time series
  static List<double> _resampleRrIntervals(
    List<double> rrIntervalsMs, {
    double samplingRate = 4.0,
  }) {
    if (rrIntervalsMs.isEmpty) return [];

    // Create cumulative time array
    final times = <double>[0.0];
    for (var rr in rrIntervalsMs) {
      times.add(times.last + rr);
    }

    // Create uniform time grid
    final dt = 1000.0 / samplingRate; // ms
    final totalDuration = times.last;
    final numSamples = (totalDuration / dt).floor();

    if (numSamples < 2) return [];

    // Interpolate RR values onto uniform grid
    final resampled = <double>[];
    for (int i = 0; i < numSamples; i++) {
      final t = i * dt;

      // Find surrounding RR intervals
      int idx = 0;
      while (idx < times.length - 1 && times[idx + 1] < t) {
        idx++;
      }

      if (idx >= rrIntervalsMs.length) break;

      // Linear interpolation
      if (idx == times.length - 1) {
        resampled.add(rrIntervalsMs[idx]);
      } else {
        final t0 = times[idx];
        final t1 = times[idx + 1];
        final rr0 = rrIntervalsMs[idx];
        final rr1 =
            idx + 1 < rrIntervalsMs.length ? rrIntervalsMs[idx + 1] : rr0;

        final alpha = (t - t0) / (t1 - t0);
        resampled.add(rr0 + alpha * (rr1 - rr0));
      }
    }

    return resampled;
  }

  /// Welch's method for Power Spectral Density estimation
  static List<double> _welchPsd(
    List<double> signal, {
    double samplingRate = 4.0,
  }) {
    // Simplified Welch's method - uses single window
    final n = signal.length;

    // Apply Hanning window
    final windowed = <double>[];
    for (int i = 0; i < n; i++) {
      final window = 0.5 * (1 - cos(2 * pi * i / (n - 1)));
      windowed.add(signal[i] * window);
    }

    // Compute FFT (simplified using DFT for small windows)
    final psd = _computePowerSpectrum(windowed);

    return psd;
  }

  /// Compute power spectrum using DFT
  static List<double> _computePowerSpectrum(List<double> signal) {
    final n = signal.length;
    final halfN = (n / 2).floor();
    final psd = List<double>.filled(halfN, 0.0);

    for (int k = 0; k < halfN; k++) {
      double real = 0.0;
      double imag = 0.0;

      for (int i = 0; i < n; i++) {
        final angle = -2 * pi * k * i / n;
        real += signal[i] * cos(angle);
        imag += signal[i] * sin(angle);
      }

      // Power = |X[k]|^2 / N
      psd[k] = (real * real + imag * imag) / n;
    }

    return psd;
  }

  /// Calculate band power from PSD
  static double _bandPower(
    List<double> psd,
    double lowFreq,
    double highFreq,
    double samplingRate,
  ) {
    final n = psd.length;
    final freqResolution = samplingRate / (2 * n);

    final lowIdx = (lowFreq / freqResolution).floor().clamp(0, n - 1);
    final highIdx = (highFreq / freqResolution).ceil().clamp(0, n - 1);

    double power = 0.0;
    for (int i = lowIdx; i <= highIdx; i++) {
      power += psd[i];
    }

    return power * freqResolution; // Multiply by frequency resolution
  }

  // ============================================================================
  // NON-LINEAR FEATURES
  // ============================================================================

  /// Compute SD1/SD2 ratio (Poincaré plot analysis)
  static double computeSd1Sd2(List<double> rrIntervalsMs) {
    if (rrIntervalsMs.length < 3) return 0.0;

    final validRr =
        rrIntervalsMs.where((rr) => rr >= 300 && rr <= 2000).toList();
    if (validRr.length < 3) return 0.0;

    // Calculate successive differences
    final diffs = <double>[];
    for (int i = 1; i < validRr.length; i++) {
      diffs.add(validRr[i] - validRr[i - 1]);
    }

    // SD1: Standard deviation perpendicular to line of identity
    // SD1 = sqrt(0.5 * var(diff))
    final meanDiff = diffs.reduce((a, b) => a + b) / diffs.length;
    final varDiff =
        diffs.map((d) => pow(d - meanDiff, 2)).reduce((a, b) => a + b) /
        diffs.length;
    final sd1 = sqrt(0.5 * varDiff);

    // SD2: Standard deviation along line of identity
    // SD2 = sqrt(2 * SDNN^2 - 0.5 * SD1^2)
    final sdnn = computeSdnn(validRr);
    final sd2Squared = 2 * sdnn * sdnn - 0.5 * sd1 * sd1;
    final sd2 = sd2Squared > 0 ? sqrt(sd2Squared) : 0.0;

    return sd2 > 0 ? (sd1 / sd2) : 0.0;
  }

  /// Compute Sample Entropy (SampEn)
  /// m = embedding dimension (default 2)
  /// r = tolerance (default 0.2 * SD of signal)
  static double computeSampleEntropy(
    List<double> rrIntervalsMs, {
    int m = 2,
    double? r,
  }) {
    final validRr =
        rrIntervalsMs.where((rr) => rr >= 300 && rr <= 2000).toList();
    if (validRr.length < m + 2) return 0.0;

    // Calculate tolerance if not provided (0.2 * SD)
    final sd = computeSdnn(validRr);
    final tolerance = r ?? (0.2 * sd);

    if (tolerance == 0) return 0.0;

    // Count template matches for m and m+1
    final countM = _countTemplateMatches(validRr, m, tolerance);
    final countM1 = _countTemplateMatches(validRr, m + 1, tolerance);

    if (countM == 0 || countM1 == 0) return 0.0;

    return -log(countM1 / countM);
  }

  /// Helper for Sample Entropy calculation
  static double _countTemplateMatches(List<double> signal, int m, double r) {
    final n = signal.length;
    int count = 0;

    for (int i = 0; i < n - m; i++) {
      for (int j = i + 1; j < n - m; j++) {
        bool match = true;
        for (int k = 0; k < m; k++) {
          if ((signal[i + k] - signal[j + k]).abs() > r) {
            match = false;
            break;
          }
        }
        if (match) count++;
      }
    }

    return count.toDouble();
  }

  /// Compute Detrended Fluctuation Analysis (DFA) alpha1
  /// Simplified implementation for short-term scaling exponent
  static double computeDfaAlpha1(List<double> rrIntervalsMs) {
    final validRr =
        rrIntervalsMs.where((rr) => rr >= 300 && rr <= 2000).toList();
    if (validRr.length < 16) return 0.0;

    // Create cumulative sum (integration)
    final mean = validRr.reduce((a, b) => a + b) / validRr.length;
    final cumSum = <double>[0.0];
    for (var rr in validRr) {
      cumSum.add(cumSum.last + (rr - mean));
    }
    cumSum.removeAt(0);

    // Define window sizes (box sizes) for short-term (4-16 beats)
    final boxSizes = [4, 6, 8, 10, 12, 14, 16];
    final logSizes = <double>[];
    final logFluctuations = <double>[];

    for (var n in boxSizes) {
      if (n > cumSum.length) break;

      // Calculate fluctuation for this box size
      final fluctuation = _dfaFluctuation(cumSum, n);
      if (fluctuation > 0) {
        logSizes.add(log(n.toDouble()));
        logFluctuations.add(log(fluctuation));
      }
    }

    if (logSizes.length < 2) return 0.0;

    // Linear regression to find slope (alpha)
    final alpha = _linearRegression(logSizes, logFluctuations);

    return alpha;
  }

  /// Calculate DFA fluctuation for a given box size
  static double _dfaFluctuation(List<double> cumSum, int boxSize) {
    final n = cumSum.length;
    final numBoxes = (n / boxSize).floor();

    double totalVariance = 0.0;

    for (int i = 0; i < numBoxes; i++) {
      final start = i * boxSize;
      final end = start + boxSize;

      // Extract segment
      final segment = cumSum.sublist(start, end);

      // Fit linear trend
      final x = List.generate(boxSize, (i) => i.toDouble());
      final slope = _linearRegression(x, segment);
      final intercept =
          segment.reduce((a, b) => a + b) / boxSize - slope * (boxSize - 1) / 2;

      // Calculate variance from trend
      for (int j = 0; j < boxSize; j++) {
        final trend = intercept + slope * j;
        final diff = segment[j] - trend;
        totalVariance += diff * diff;
      }
    }

    return sqrt(totalVariance / (numBoxes * boxSize));
  }

  /// Simple linear regression (returns slope)
  static double _linearRegression(List<double> x, List<double> y) {
    if (x.length != y.length || x.isEmpty) return 0.0;

    final n = x.length;
    final meanX = x.reduce((a, b) => a + b) / n;
    final meanY = y.reduce((a, b) => a + b) / n;

    double numerator = 0.0;
    double denominator = 0.0;

    for (int i = 0; i < n; i++) {
      numerator += (x[i] - meanX) * (y[i] - meanY);
      denominator += (x[i] - meanX) * (x[i] - meanX);
    }

    return denominator != 0 ? (numerator / denominator) : 0.0;
  }

  // ============================================================================
  // MAIN FEATURE EXTRACTION
  // ============================================================================

  /// Extract all 14 features required for WESAD model
  /// Returns features in exact order expected by model
  ///
  /// [meanHr] is optional - if provided, will be used as the HR feature instead
  /// of computing it from mean RR interval. This allows using actual HR values
  /// from the wearable device instead of deriving them from RR intervals.
  static List<double> extractAllFeatures(
    List<double> rrIntervalsMs, {
    double? meanHr,
  }) {
    final validRr =
        rrIntervalsMs.where((rr) => rr >= 300 && rr <= 2000).toList();

    if (validRr.length < 10) {
      // Return zeros if insufficient data
      return List<double>.filled(14, 0.0);
    }

    // Time domain features
    final rmssd = computeRmssd(validRr);
    final meanRr = computeMeanRr(validRr);
    final sdnn = computeSdnn(validRr);
    final pnn50 = computePnn50(validRr);

    // Frequency domain features
    final freqFeatures = computeFrequencyDomain(validRr);

    // Non-linear features
    final sd1sd2 = computeSd1Sd2(validRr);
    final sampEn = computeSampleEntropy(validRr);
    final dfaAlpha1 = computeDfaAlpha1(validRr);

    // Heart rate: use provided meanHr if available, otherwise compute from mean RR
    final hr = meanHr ?? (meanRr > 0 ? (60000.0 / meanRr) : 0.0);

    // Return in exact order: ['RMSSD', 'Mean_RR', 'HRV_SDNN', 'pNN50',
    // 'HRV_HF', 'HRV_LF', 'HRV_HF_nu', 'HRV_LF_nu', 'HRV_LFHF', 'HRV_TP',
    // 'HRV_SD1SD2', 'HRV_Sampen', 'HRV_DFA_alpha1', 'HR']
    return [
      rmssd,
      meanRr,
      sdnn,
      pnn50,
      freqFeatures['HF']!,
      freqFeatures['LF']!,
      freqFeatures['HF_nu']!,
      freqFeatures['LF_nu']!,
      freqFeatures['LFHF']!,
      freqFeatures['TP']!,
      sd1sd2,
      sampEn,
      dfaAlpha1,
      hr,
    ];
  }

  /// Convert heart rate samples to RR intervals
  static List<double> hrToRrIntervals(List<double> hrBpm) {
    return hrBpm.map((hr) => hr > 0 ? (60000.0 / hr) : 0.0).toList();
  }
}
