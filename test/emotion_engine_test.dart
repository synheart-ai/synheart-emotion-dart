import 'package:flutter_test/flutter_test.dart';
import 'package:synheart_emotion/synheart_emotion.dart';

void main() {
  group('FeatureExtractor', () {
    test('extractHrMean calculates correct mean', () {
      final hrValues = [70.0, 72.0, 68.0, 75.0];
      final mean = FeatureExtractor.extractHrMean(hrValues);
      expect(mean, equals(71.25));
    });

    test('extractHrMean handles empty list', () {
      final mean = FeatureExtractor.extractHrMean([]);
      expect(mean, equals(0.0));
    });

    test('extractSdnn calculates standard deviation', () {
      // Test with known RR intervals
      final rrIntervals = [800.0, 820.0, 810.0, 830.0, 815.0];
      final sdnn = FeatureExtractor.extractSdnn(rrIntervals);
      expect(sdnn, greaterThan(0));
      expect(sdnn.isFinite, isTrue);
    });

    test('extractRmssd calculates root mean square', () {
      final rrIntervals = [800.0, 820.0, 810.0, 830.0, 815.0];
      final rmssd = FeatureExtractor.extractRmssd(rrIntervals);
      expect(rmssd, greaterThan(0));
      expect(rmssd.isFinite, isTrue);
    });

    test('extractFeatures combines all features', () {
      final hrValues = [70.0, 72.0];
      final rrIntervals = [800.0, 820.0, 810.0];
      final motion = {'steps': 100.0};

      final features = FeatureExtractor.extractFeatures(
        hrValues: hrValues,
        rrIntervalsMs: rrIntervals,
        motion: motion,
      );

      expect(features.containsKey('hr_mean'), isTrue);
      expect(features.containsKey('sdnn'), isTrue);
      expect(features.containsKey('rmssd'), isTrue);
      expect(features.containsKey('steps'), isTrue);
    });

    test('extract14Features returns all 14 HRV features', () {
      final hrValues = [70.0, 72.0, 68.0];
      final rrIntervals = List.generate(100, (i) => 800.0 + (i % 50));

      final features = FeatureExtractor.extract14Features(
        hrValues: hrValues,
        rrIntervalsMs: rrIntervals,
      );

      // Verify all 14 features are present
      expect(features.length, equals(14));
      expect(features[0], greaterThan(0)); // RMSSD
      expect(features[1], greaterThan(0)); // Mean_RR
      expect(features[13], greaterThan(0)); // HR
    });

    test('normalizeFeatures applies z-score normalization', () {
      final features = {'hr_mean': 80.0, 'sdnn': 50.0};
      final mu = {'hr_mean': 70.0, 'sdnn': 40.0};
      final sigma = {'hr_mean': 10.0, 'sdnn': 5.0};

      final normalized = FeatureExtractor.normalizeFeatures(
        features,
        mu,
        sigma,
      );

      expect(normalized['hr_mean'], equals(1.0)); // (80-70)/10
      expect(normalized['sdnn'], equals(2.0)); // (50-40)/5
    });
  });

  // Note: LinearSvmModel tests removed in v0.2.0
  // Package now uses OnnxEmotionModel exclusively
  // See CHANGELOG.md for migration guide

  group('EmotionEngine', () {
    late EmotionEngine engine;

    setUp(() {
      engine = EmotionEngine.fromPretrained(
        const EmotionConfig(
          window: Duration(seconds: 10),
          step: Duration(seconds: 1),
          minRrCount: 5,
        ),
      );
    });

    test('push adds data to buffer', () {
      final statsBefore = engine.getBufferStats();
      expect(statsBefore['count'], equals(0));

      engine.push(
        hr: 70.0,
        rrIntervalsMs: [800.0, 820.0, 810.0],
        timestamp: DateTime.now().toUtc(),
      );

      final statsAfter = engine.getBufferStats();
      expect(statsAfter['count'], equals(1));
    });

    test('consumeReady returns empty list when not enough data', () {
      engine.push(
        hr: 70.0,
        rrIntervalsMs: [800.0], // Only 1 RR interval, need 5
        timestamp: DateTime.now().toUtc(),
      );

      final results = engine.consumeReady();
      expect(results, isEmpty);
    });

    test('consumeReadyAsync returns empty list when not enough data', () async {
      engine.push(
        hr: 70.0,
        rrIntervalsMs: [800.0], // Only 1 RR interval, need 5
        timestamp: DateTime.now().toUtc(),
      );

      final results = await engine.consumeReadyAsync();
      expect(results, isEmpty);
    });

    test('consumeReadyAsync returns results when enough data', () async {
      // Create a mock model for testing
      final mockModel = _MockEmotionModel();
      final engineWithModel = EmotionEngine.fromPretrained(
        const EmotionConfig(
          window: Duration(seconds: 10),
          step: Duration(seconds: 1),
          minRrCount: 5,
        ),
        model: mockModel,
      );

      // Add enough data points
      for (int i = 0; i < 3; i++) {
        engineWithModel.push(
          hr: 70.0 + i,
          rrIntervalsMs: [800.0, 820.0, 810.0, 830.0, 815.0],
          timestamp: DateTime.now().toUtc().subtract(Duration(seconds: i)),
        );
      }

      final results = await engineWithModel.consumeReadyAsync();
      expect(results, isNotEmpty);

      final result = results.first;
      expect(result.emotion, isIn(['Baseline', 'Stress']));
      expect(result.confidence, greaterThan(0.0));
      expect(result.confidence, lessThanOrEqualTo(1.0));
    });

    test('clear removes all buffered data', () {
      engine.push(
        hr: 70.0,
        rrIntervalsMs: [800.0, 820.0, 810.0],
        timestamp: DateTime.now().toUtc(),
      );

      expect(engine.getBufferStats()['count'], greaterThan(0));

      engine.clear();

      expect(engine.getBufferStats()['count'], equals(0));
    });
  });

  group('EmotionResult', () {
    test('fromInference creates result with correct top emotion', () {
      final probabilities = {'Baseline': 0.6, 'Stress': 0.4};
      final features = {'hr_mean': 70.0, 'sdnn': 40.0};
      final model = {'id': 'test', 'version': '1.0'};

      final result = EmotionResult.fromInference(
        timestamp: DateTime.now(),
        probabilities: probabilities,
        features: features,
        model: model,
      );

      expect(result.emotion, equals('Baseline'));
      expect(result.confidence, equals(0.6));
      expect(result.probabilities, equals(probabilities));
    });

    test('toJson and fromJson round trip correctly', () {
      final original = EmotionResult.fromInference(
        timestamp: DateTime(2023, 1, 1, 12, 0, 0),
        probabilities: {'Baseline': 0.8, 'Stress': 0.2},
        features: {'hr_mean': 70.0},
        model: {'id': 'test'},
      );

      final json = original.toJson();
      final restored = EmotionResult.fromJson(json);

      expect(restored.timestamp, equals(original.timestamp));
      expect(restored.emotion, equals(original.emotion));
      expect(restored.confidence, equals(original.confidence));
    });
  });

  group('EmotionConfig', () {
    test('default values are correct', () {
      const config = EmotionConfig();

      expect(
        config.modelId,
        equals('extratrees_chest_ecg_w120s60_binary_v1_0'),
      );
      expect(config.window, equals(Duration(seconds: 120)));
      expect(config.step, equals(Duration(seconds: 60)));
      expect(config.minRrCount, equals(30));
      expect(config.returnAllProbas, isTrue);
    });

    test('copyWith creates modified copy', () {
      const original = EmotionConfig();
      final modified = original.copyWith(
        window: Duration(seconds: 30),
        minRrCount: 20,
      );

      expect(modified.window, equals(Duration(seconds: 30)));
      expect(modified.minRrCount, equals(20));
      expect(modified.step, equals(original.step)); // Unchanged
    });
  });

  group('EmotionError', () {
    test('tooFewRR error has correct message', () {
      final error = EmotionError.tooFewRR(minExpected: 30, actual: 10);
      expect(error.message, contains('Too few RR intervals'));
      expect(error.message, contains('30'));
      expect(error.message, contains('10'));
    });

    test('badInput error has correct message', () {
      final error = EmotionError.badInput('Invalid HR value');
      expect(error.message, contains('Bad input'));
      expect(error.message, contains('Invalid HR value'));
    });
  });
}

/// Mock model for testing EmotionEngine without requiring actual ONNX model
class _MockEmotionModel {
  Map<String, double> predict(Map<String, double> features) {
    // Return mock probabilities for binary classification (Baseline/Stress)
    return {'Baseline': 0.6, 'Stress': 0.4};
  }

  Future<Map<String, double>> predictAsync(Map<String, double> features) async {
    return predict(features);
  }

  Map<String, dynamic> getMetadata() {
    return {
      'id': 'mock_model',
      'version': '1.0',
      'model_id': 'extratrees_chest_ecg_w120s60_binary_v1_0',
    };
  }

  // Mock ONNX model properties
  List<String> get inputNames => List.generate(14, (i) => 'feature_$i');
  List<String> get classNames => ['Baseline', 'Stress'];

  @override
  String toString() => 'MockEmotionModel';
}
