import 'package:flutter/material.dart';
import 'package:synheart_emotion/synheart_emotion.dart';

void main() {
  runApp(const EmotionApp());
}

class EmotionApp extends StatelessWidget {
  const EmotionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Synheart Emotion Detection',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: const EmotionDetectionPage(),
    );
  }
}

class EmotionDetectionPage extends StatefulWidget {
  const EmotionDetectionPage({super.key});

  @override
  State<EmotionDetectionPage> createState() => _EmotionDetectionPageState();
}

class _EmotionDetectionPageState extends State<EmotionDetectionPage> {
  late EmotionEngine _engine;
  EmotionResult? _latestResult;
  bool _isLoading = false;
  String _statusMessage = 'Ready to detect emotions';

  @override
  void initState() {
    super.initState();
    _initializeEngine();
  }

  void _initializeEngine() {
    _engine = EmotionEngine.fromPretrained(
      const EmotionConfig(
        window: Duration(seconds: 60),
        step: Duration(seconds: 5),
        minRrCount: 30,
      ),
    );
  }

  Future<void> _simulateDataAndDetect() async {
    setState(() {
      _isLoading = true;
      _statusMessage = 'Collecting biometric data...';
    });

    // Simulate pushing biometric data
    // In a real app, this would come from a wearable device or health sensor
    final now = DateTime.now().toUtc();
    for (int i = 0; i < 20; i++) {
      _engine.push(
        hr: 70.0 + (i * 0.5), // Simulated heart rate
        rrIntervalsMs: List.generate(
          10, // More RR intervals per data point
          (j) => 800.0 + (j * 10.0) + (i * 2.0),
        ), // Simulated RR intervals
        timestamp: now.subtract(Duration(seconds: 20 - i)),
      );
    }

    // Wait a bit for processing
    await Future.delayed(const Duration(milliseconds: 500));

    // Consume ready results
    final results = await _engine.consumeReady();

    setState(() {
      _isLoading = false;
      if (results.isNotEmpty) {
        _latestResult = results.first;
        _statusMessage = 'Emotion detected successfully';
      } else {
        _statusMessage = 'Not enough data yet. Need more RR intervals.';
      }
    });
  }

  void _clearData() {
    _engine.clear();
    setState(() {
      _latestResult = null;
      _statusMessage = 'Data cleared. Ready to detect emotions.';
    });
  }

  Color _getEmotionColor(String emotion) {
    switch (emotion.toLowerCase()) {
      case 'calm':
        return Colors.green;
      case 'stressed':
        return Colors.red;
      case 'amused':
        return Colors.orange;
      default:
        return Colors.blue;
    }
  }

  IconData _getEmotionIcon(String emotion) {
    switch (emotion.toLowerCase()) {
      case 'calm':
        return Icons.spa;
      case 'stressed':
        return Icons.warning;
      case 'amused':
        return Icons.mood;
      default:
        return Icons.favorite;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Emotion Detection'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        elevation: 0,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Status message
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Row(
                  children: [
                    if (_isLoading)
                      const Padding(
                        padding: EdgeInsets.only(right: 12.0),
                        child: SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        ),
                      ),
                    Expanded(
                      child: Text(
                        _statusMessage,
                        style: Theme.of(context).textTheme.bodyLarge,
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),

            // Main emotion display
            if (_latestResult != null) ...[
              Card(
                elevation: 4,
                child: Container(
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(12),
                    gradient: LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: [
                        _getEmotionColor(
                          _latestResult!.emotion,
                        ).withValues(alpha: 0.1),
                        _getEmotionColor(
                          _latestResult!.emotion,
                        ).withValues(alpha: 0.05),
                      ],
                    ),
                  ),
                  padding: const EdgeInsets.all(24.0),
                  child: Column(
                    children: [
                      Icon(
                        _getEmotionIcon(_latestResult!.emotion),
                        size: 64,
                        color: _getEmotionColor(_latestResult!.emotion),
                      ),
                      const SizedBox(height: 16),
                      Text(
                        _latestResult!.emotion,
                        style: Theme.of(
                          context,
                        ).textTheme.displayMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: _getEmotionColor(_latestResult!.emotion),
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        '${(_latestResult!.confidence * 100).toStringAsFixed(1)}% confidence',
                        style: Theme.of(context).textTheme.titleMedium
                            ?.copyWith(color: Colors.grey[600]),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 24),

              // Probability breakdown
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Emotion Probabilities',
                        style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 16),
                      ..._latestResult!.probabilities.entries.map((entry) {
                        final isTop = entry.key == _latestResult!.emotion;
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 12.0),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceBetween,
                                children: [
                                  Row(
                                    children: [
                                      if (isTop)
                                        Icon(
                                          Icons.star,
                                          size: 16,
                                          color: _getEmotionColor(entry.key),
                                        ),
                                      const SizedBox(width: 4),
                                      Text(
                                        entry.key,
                                        style: TextStyle(
                                          fontWeight:
                                              isTop
                                                  ? FontWeight.bold
                                                  : FontWeight.normal,
                                          fontSize: 16,
                                        ),
                                      ),
                                    ],
                                  ),
                                  Text(
                                    '${(entry.value * 100).toStringAsFixed(1)}%',
                                    style: TextStyle(
                                      fontWeight:
                                          isTop
                                              ? FontWeight.bold
                                              : FontWeight.normal,
                                    ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 4),
                              ClipRRect(
                                borderRadius: BorderRadius.circular(4),
                                child: LinearProgressIndicator(
                                  value: entry.value,
                                  minHeight: 8,
                                  backgroundColor: Colors.grey[200],
                                  valueColor: AlwaysStoppedAnimation<Color>(
                                    _getEmotionColor(entry.key),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        );
                      }),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 24),

              // Features display
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Extracted Features',
                        style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 16),
                      ..._latestResult!.features.entries.map((entry) {
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 8.0),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Text(
                                entry.key,
                                style: const TextStyle(fontSize: 14),
                              ),
                              Text(
                                entry.value.toStringAsFixed(2),
                                style: TextStyle(
                                  fontSize: 14,
                                  fontWeight: FontWeight.w500,
                                  color: Colors.grey[700],
                                ),
                              ),
                            ],
                          ),
                        );
                      }),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 24),

              // Timestamp
              Text(
                'Detected at: ${_latestResult!.timestamp.toLocal().toString().substring(0, 19)}',
                style: Theme.of(
                  context,
                ).textTheme.bodySmall?.copyWith(color: Colors.grey[600]),
                textAlign: TextAlign.center,
              ),
            ] else ...[
              // Empty state
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(48.0),
                  child: Column(
                    children: [
                      Icon(Icons.psychology, size: 64, color: Colors.grey[400]),
                      const SizedBox(height: 16),
                      Text(
                        'No emotion data yet',
                        style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          color: Colors.grey[600],
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Tap the button below to simulate biometric data and detect emotions',
                        style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                          color: Colors.grey[500],
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
              ),
            ],

            const SizedBox(height: 24),

            // Action buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isLoading ? null : _simulateDataAndDetect,
                    icon: const Icon(Icons.play_arrow),
                    label: const Text('Detect Emotion'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                  ),
                ),
                if (_latestResult != null) ...[
                  const SizedBox(width: 12),
                  ElevatedButton.icon(
                    onPressed: _clearData,
                    icon: const Icon(Icons.clear),
                    label: const Text('Clear'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      backgroundColor: Colors.grey[300],
                      foregroundColor: Colors.black87,
                    ),
                  ),
                ],
              ],
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _engine.clear();
    super.dispose();
  }
}
