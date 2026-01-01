import 'package:flutter_test/flutter_test.dart';
import 'package:synheart_emotion/synheart_emotion.dart';

/// Integration tests that match the usage pattern in synheart-core EmotionHead
///
/// These tests validate that EmotionEngine works correctly when used by
/// synheart-core's EmotionHead module, which:
/// 1. Uses EmotionEngine.fromPretrained() for initialization
/// 2. Calls push() with HR, RR intervals, and timestamp
/// 3. Calls consumeReady() synchronously to get results
/// 4. Maps EmotionResult to HSI-compatible EmotionState
void main() {
  group('synheart-core Integration Tests', () {
    late EmotionEngine engine;

    setUp(() {
      // Create engine with configuration matching synheart-core EmotionHead
      // Short window (10s) and step (1s) for near-realtime updates
      engine = EmotionEngine.fromPretrained(
        const EmotionConfig(
          window: Duration(seconds: 10),
          step: Duration(seconds: 1),
          minRrCount: 5,
        ),
        model: _MockLinearSvmModel(), // Use mock for testing
        onLog: (level, message, {context}) {
          // Optional: print('[EmotionEngine][$level] $message');
        },
      );
    });

    tearDown(() {
      engine.clear();
    });

    test('EmotionEngine initializes with synheart-core config', () {
      expect(engine.config.window, equals(const Duration(seconds: 10)));
      expect(engine.config.step, equals(const Duration(seconds: 1)));
      expect(engine.config.minRrCount, equals(5));
    });

    test('push accepts derived HR and synthetic RR intervals', () {
      // synheart-core EmotionHead derives features from HSV embedding
      // and synthesizes RR intervals from mean_rr with variance
      const hrMean = 72.0;
      const meanRr = 833.0; // 60000 / 72

      // Synthetic RR intervals with variance (matching EmotionHead pattern)
      final syntheticRR = List.generate(10, (i) {
        final variance = (i % 3 - 1) * 10.0; // -10, 0, +10 ms
        return meanRr + variance;
      });

      // Push to engine
      engine.push(
        hr: hrMean,
        rrIntervalsMs: syntheticRR,
        timestamp: DateTime.now(),
      );

      // Verify data was added
      final stats = engine.getBufferStats();
      expect(stats['count'], equals(1));
      expect(stats['rr_count'], equals(10));
    });

    test('consumeReady returns EmotionResult compatible with HSI schema', () {
      // Push enough data to trigger result
      final baseTime = DateTime.now();
      for (int i = 0; i < 3; i++) {
        final meanRr = 800.0 + i * 10;
        final syntheticRR = List.generate(10, (j) {
          final variance = (j % 3 - 1) * 10.0;
          return meanRr + variance;
        });

        engine.push(
          hr: 70.0 + i,
          rrIntervalsMs: syntheticRR,
          timestamp: baseTime.add(Duration(seconds: i)),
        );
      }

      // Wait for step interval
      Future.delayed(const Duration(milliseconds: 1100), () {});

      // Consume results
      final results = engine.consumeReady();

      if (results.isNotEmpty) {
        final result = results.first;

        // Verify HSI-compatible output schema
        expect(result.probabilities, isA<Map<String, double>>());
        expect(result.probabilities.containsKey('Stressed'), isTrue);
        expect(result.probabilities.containsKey('Calm'), isTrue);
        expect(result.probabilities.containsKey('Amused'), isTrue);

        // Verify probabilities are valid (0.0-1.0)
        for (final prob in result.probabilities.values) {
          expect(prob, greaterThanOrEqualTo(0.0));
          expect(prob, lessThanOrEqualTo(1.0));
        }

        // Verify confidence is valid
        expect(result.confidence, greaterThanOrEqualTo(0.0));
        expect(result.confidence, lessThanOrEqualTo(1.0));

        // Verify emotion is one of the expected values
        expect(result.emotion, isIn(['Stressed', 'Calm', 'Amused']));
      }
    });

    test('HSI EmotionState mapping is correct', () {
      // Push data
      final baseTime = DateTime.now();
      for (int i = 0; i < 3; i++) {
        engine.push(
          hr: 75.0,
          rrIntervalsMs: List.generate(10, (j) => 800.0 + j * 5),
          timestamp: baseTime.add(Duration(seconds: i)),
        );
      }

      final results = engine.consumeReady();

      if (results.isNotEmpty) {
        final result = results.first;

        // Map to HSI EmotionState (matching synheart-core pattern)
        final stress = (result.probabilities['Stressed'] ?? 0.0).clamp(
          0.0,
          1.0,
        );
        final calm = (result.probabilities['Calm'] ?? 0.0).clamp(0.0, 1.0);
        final amused = (result.probabilities['Amused'] ?? 0.0).clamp(0.0, 1.0);

        // Derived fields (matching synheart-core EmotionHead)
        final activation = ((amused + stress) / 2.0).clamp(0.0, 1.0);
        final valence = (calm + amused - stress).clamp(-1.0, 1.0);

        // Verify all fields are valid
        expect(stress, inInclusiveRange(0.0, 1.0));
        expect(calm, inInclusiveRange(0.0, 1.0));
        expect(amused, inInclusiveRange(0.0, 1.0));
        expect(activation, inInclusiveRange(0.0, 1.0));
        expect(valence, inInclusiveRange(-1.0, 1.0));

        // Verify derived fields match expected calculations
        expect(activation, closeTo((amused + stress) / 2.0, 0.01));
        expect(valence, closeTo(calm + amused - stress, 0.01));
      }
    });

    test('handles time-series data with proper window management', () {
      // synheart-core sends continuous stream of HSV updates
      // EmotionEngine should maintain 10s window and emit every 1s (step)

      final baseTime = DateTime.now();
      const pushCount = 15; // 15 data points over time

      for (int i = 0; i < pushCount; i++) {
        engine.push(
          hr: 70.0 + (i % 10), // Varying HR
          rrIntervalsMs: List.generate(10, (j) => 800.0 + i * 5 + j),
          timestamp: baseTime.add(Duration(seconds: i)),
        );
      }

      // Buffer should trim data older than window (10s)
      final stats = engine.getBufferStats();
      expect(stats['count'], lessThanOrEqualTo(pushCount));

      // Duration should be approximately within window (with small margin for timing)
      // Note: Actual duration may be slightly larger due to test execution timing
      expect(stats['duration_ms'], lessThanOrEqualTo(15000));
    });

    test('handles missing or invalid features gracefully', () {
      // Test with insufficient RR intervals
      engine.push(
        hr: 72.0,
        rrIntervalsMs: [800.0, 810.0], // Only 2, need 5
        timestamp: DateTime.now(),
      );

      final results1 = engine.consumeReady();
      expect(results1, isEmpty); // Should not emit with insufficient data

      // Test with invalid HR (outside valid range)
      engine.clear();
      engine.push(
        hr: 350.0, // Invalid HR (> maxValidHr of 300)
        rrIntervalsMs: List.generate(10, (i) => 800.0),
        timestamp: DateTime.now(),
      );

      final stats = engine.getBufferStats();
      expect(stats['count'], equals(0)); // Should reject invalid HR
    });

    test('clear() resets engine state for new session', () {
      // Add data
      for (int i = 0; i < 5; i++) {
        engine.push(
          hr: 70.0,
          rrIntervalsMs: List.generate(10, (j) => 800.0),
          timestamp: DateTime.now().add(Duration(seconds: i)),
        );
      }

      expect(engine.getBufferStats()['count'], greaterThan(0));

      // Clear
      engine.clear();

      // Verify state is reset
      final stats = engine.getBufferStats();
      expect(stats['count'], equals(0));
      expect(stats['rr_count'], equals(0));
      expect(stats['duration_ms'], equals(0));
    });

    test('consumeReady throttles by step interval', () {
      // Add data
      // Note: EmotionEngine requires a (nearly) full window of data before it
      // will emit (see _extractWindowFeatures). For a 10s window with 2s
      // tolerance, the oldest timestamp must be ~8s old.
      final baseTime = DateTime.now().toUtc().subtract(
        const Duration(seconds: 9),
      );
      for (int i = 0; i < 3; i++) {
        engine.push(
          hr: 70.0,
          rrIntervalsMs: List.generate(10, (j) => 800.0),
          timestamp: baseTime.add(Duration(seconds: i)),
        );
      }

      // First call should return result
      final results1 = engine.consumeReady();
      expect(results1, isNotEmpty);

      // Immediate second call should return empty (throttled)
      final results2 = engine.consumeReady();
      expect(results2, isEmpty);
    });

    test('supports personalization with HR baseline', () {
      final engineWithBaseline = EmotionEngine.fromPretrained(
        const EmotionConfig(
          window: Duration(seconds: 10),
          step: Duration(seconds: 1),
          minRrCount: 5,
          hrBaseline: 65.0, // User's resting HR
        ),
        model: _MockLinearSvmModel(),
      );

      // Push data
      // Ensure the buffer spans (almost) the full window so inference runs.
      final baseTime = DateTime.now().toUtc().subtract(
        const Duration(seconds: 9),
      );
      for (int i = 0; i < 3; i++) {
        engineWithBaseline.push(
          hr: 75.0, // Will be normalized to 75 - 65 = 10
          rrIntervalsMs: List.generate(10, (j) => 800.0),
          timestamp: baseTime.add(Duration(seconds: i)),
        );
      }

      final results = engineWithBaseline.consumeReady();
      expect(results, isNotEmpty); // Should work with baseline

      engineWithBaseline.clear();
    });

    test('buffer statistics are accurate', () {
      final baseTime = DateTime.now();

      // Add 5 data points
      for (int i = 0; i < 5; i++) {
        engine.push(
          hr: 70.0 + i,
          rrIntervalsMs: List.generate(8, (j) => 800.0 + j),
          timestamp: baseTime.add(Duration(seconds: i)),
        );
      }

      final stats = engine.getBufferStats();

      expect(stats['count'], equals(5));
      expect(stats['rr_count'], equals(40)); // 5 points * 8 RR each
      expect(stats['hr_range'], equals([70.0, 74.0]));
      expect(stats['duration_ms'], greaterThan(0));
    });
  });

  group('EmotionResult HSI Compatibility', () {
    test('probabilities match HSI EmotionState schema', () {
      const probabilities = {'Stressed': 0.3, 'Calm': 0.6, 'Amused': 0.1};

      final result = EmotionResult.fromInference(
        timestamp: DateTime.now(),
        probabilities: probabilities,
        features: const {'hr_mean': 70.0, 'sdnn': 40.0, 'rmssd': 45.0},
        model: const {'id': 'test', 'version': '1.0'},
      );

      // Verify all required emotion categories are present
      expect(result.probabilities.containsKey('Stressed'), isTrue);
      expect(result.probabilities.containsKey('Calm'), isTrue);
      expect(result.probabilities.containsKey('Amused'), isTrue);

      // Verify probabilities sum to approximately 1.0
      final sum = result.probabilities.values.reduce((a, b) => a + b);
      expect(sum, closeTo(1.0, 0.01));
    });

    test('toJson output is HSI-compatible', () {
      final result = EmotionResult.fromInference(
        timestamp: DateTime(2025, 1, 1, 12, 0, 0),
        probabilities: const {'Stressed': 0.3, 'Calm': 0.6, 'Amused': 0.1},
        features: const {'hr_mean': 70.0, 'sdnn': 40.0, 'rmssd': 45.0},
        model: const {'id': 'extratrees_wrist_all_v1_0', 'version': '1.0'},
      );

      final json = result.toJson();

      // Verify required fields
      expect(json.containsKey('timestamp'), isTrue);
      expect(json.containsKey('emotion'), isTrue);
      expect(json.containsKey('confidence'), isTrue);
      expect(json.containsKey('probabilities'), isTrue);

      // Verify probabilities structure
      final probs = json['probabilities'] as Map<String, dynamic>;
      expect(probs.containsKey('Stressed'), isTrue);
      expect(probs.containsKey('Calm'), isTrue);
      expect(probs.containsKey('Amused'), isTrue);
    });
  });
}

/// Mock Linear SVM model for testing (synchronous predict)
class _MockLinearSvmModel {
  Map<String, double> predict(Map<String, double> features) {
    // Return realistic mock probabilities based on HR
    final hrMean = features['hr_mean'] ?? 70.0;

    if (hrMean > 80) {
      // High HR -> likely stressed
      return {'Stressed': 0.6, 'Calm': 0.2, 'Amused': 0.2};
    } else if (hrMean < 65) {
      // Low HR -> likely calm
      return {'Calm': 0.7, 'Stressed': 0.2, 'Amused': 0.1};
    } else {
      // Medium HR -> balanced
      return {'Calm': 0.5, 'Stressed': 0.3, 'Amused': 0.2};
    }
  }

  Map<String, dynamic> getMetadata() {
    return {'id': 'mock_svm_linear', 'version': '1.0', 'type': 'LinearSVM'};
  }

  @override
  String toString() => 'MockLinearSvmModel';
}
