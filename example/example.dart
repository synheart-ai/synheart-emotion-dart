// ignore_for_file: avoid_print, avoid_redundant_argument_values, omit_local_variable_types

import 'package:synheart_emotion/synheart_emotion.dart';

/// Example demonstrating basic usage of synheart_emotion package.
///
/// This example shows how to:
/// 1. Initialize an EmotionEngine
/// 2. Push biometric data (HR and RR intervals)
/// 3. Consume emotion results
void main() async {
  // Initialize the emotion engine with default configuration
  final engine = EmotionEngine.fromPretrained(
    const EmotionConfig(
      window: Duration(seconds: 60),
      step: Duration(seconds: 5),
      minRrCount: 30,
    ),
  );

  // Simulate pushing biometric data
  // In a real app, this would come from a wearable device or health sensor
  for (int i = 0; i < 10; i++) {
    engine.push(
      hr: 70.0 + (i * 0.5), // Simulated heart rate
      rrIntervalsMs: List.generate(
        5,
        (j) => 800.0 + (j * 10.0) + (i * 2.0),
      ), // Simulated RR intervals
      timestamp: DateTime.now().toUtc().subtract(Duration(seconds: 10 - i)),
    );
  }

  // Consume ready results
  final results = await engine.consumeReady();

  // Display results
  if (results.isNotEmpty) {
    final result = results.first;
    print('Detected emotion: ${result.emotion}');
    print('Confidence: ${(result.confidence * 100).toStringAsFixed(1)}%');
    print('Probabilities:');
    result.probabilities.forEach((emotion, prob) {
      print('  $emotion: ${(prob * 100).toStringAsFixed(1)}%');
    });
  } else {
    print('Not enough data yet. Need more RR intervals.');
  }

  // Clean up
  engine.clear();
}
