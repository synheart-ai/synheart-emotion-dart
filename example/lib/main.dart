import 'dart:async';
import 'dart:math';
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
  EmotionEngine? _engine;
  EmotionResult? _latestResult;
  bool _isRunning = false;
  bool _isInitializing = false;
  bool _isInitialized = false;
  String _statusMessage = 'Initializing...';
  String _modelName = 'Loading...';
  Timer? _dataTimer;
  Timer? _inferenceTimer;
  int _dataCollectionSeconds = 0;

  @override
  void initState() {
    super.initState();
    _initializeEngine();
  }

  Future<void> _initializeEngine() async {
    setState(() {
      _isInitializing = true;
      _statusMessage = 'Loading 60_5 model...';
    });

    try {
      // Load the 60_5 ONNX model - ExtraTrees_60_5_nozipmap.onnx
      final onnxModel = await OnnxEmotionModel.loadFromAsset(
        modelAssetPath: 'assets/ml/ExtraTrees_60_5_nozipmap.onnx',
      );

      // Create engine with the loaded model
      // Use the modelId from the loaded model (read from metadata)
      _engine = EmotionEngine.fromPretrained(
        EmotionConfig(
          modelId: onnxModel.modelId, // Use actual modelId from loaded model
          window: const Duration(seconds: 60),
          step: const Duration(seconds: 5),
          minRrCount: 30,
        ),
        model: onnxModel,
      );

      setState(() {
        _isInitialized = true;
        _statusMessage = 'Ready to detect emotions';
        _modelName = 'ExtraTrees 60s/5s (${onnxModel.modelId})';
      });
    } catch (e) {
      setState(() {
        _isInitializing = false;
        _statusMessage = 'Error loading model: $e';
      });
    }
  }

  void _startSimulation() {
    if (_isRunning || !_isInitialized || _engine == null) return;

    setState(() {
      _isRunning = true;
      _dataCollectionSeconds = 0;
      _statusMessage = 'Collecting data... 0s';
    });

    // Simulate data every 500ms (like the working example)
    _dataTimer = Timer.periodic(const Duration(milliseconds: 500), (timer) {
      _simulateDataPoint();
      setState(() {
        // Increment by 0.5 seconds each time (500ms = 0.5s)
        _dataCollectionSeconds++;
        final seconds = _dataCollectionSeconds / 2.0; // Convert to seconds
        if (_latestResult == null) {
          _statusMessage = 'Collecting data... ${seconds.toStringAsFixed(1)}s';
        }
      });
    });

    // Run inference based on step size (5 seconds)
    _inferenceTimer = Timer.periodic(const Duration(seconds: 5), (_) {
      _runInference();
    });
  }

  void _stopSimulation() {
    _dataTimer?.cancel();
    _inferenceTimer?.cancel();

    setState(() {
      _isRunning = false;
      _statusMessage = 'Stopped. Ready to detect emotions.';
    });
  }

  void _simulateDataPoint() {
    final random = Random();

    // Simulate HR from watch (input is HR only)
    final baseHr = 70 + (random.nextDouble() - 0.5) * 20; // ~70 BPM ± 10
    final hr = baseHr.clamp(50.0, 120.0);

    // Convert HR to RR intervals using pushFromHrSamples
    // Generate multiple HR samples to get more RR intervals
    // This simulates having multiple HR readings from the watch
    final hrSamples = <double>[];
    for (int i = 0; i < 10; i++) {
      // Add small variation to HR samples
      final hrSample = hr + (random.nextDouble() - 0.5) * 5.0;
      hrSamples.add(hrSample.clamp(50.0, 120.0));
    }

    // Use pushFromHrSamples to convert HR to RR automatically
    _engine!.pushFromHrSamples(
      hrSamples: hrSamples,
      timestamp: DateTime.now().toUtc(),
    );
  }

  void _runInference() async {
    if (_engine == null) return;

    try {
      final results = await _engine!.consumeReadyAsync();

      if (results.isNotEmpty) {
        setState(() {
          _latestResult = results.first;
          _statusMessage =
              'Emotion detected: ${results.first.emotion} (${(results.first.confidence * 100).toStringAsFixed(1)}%)';
        });
      }
    } catch (e) {
      setState(() {
        _statusMessage = 'Inference error: $e';
      });
    }
  }

  void _clearData() {
    _stopSimulation();
    _engine?.clear();
    setState(() {
      _latestResult = null;
      _dataCollectionSeconds = 0;
      _statusMessage = 'Data cleared. Ready to detect emotions.';
    });
  }

  Color _getEmotionColor(String emotion) {
    switch (emotion.toLowerCase()) {
      case 'baseline':
        return Colors.green;
      case 'stress':
        return Colors.red;
      default:
        return Colors.blue;
    }
  }

  IconData _getEmotionIcon(String emotion) {
    switch (emotion.toLowerCase()) {
      case 'baseline':
        return Icons.spa;
      case 'stress':
        return Icons.warning;
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
            // Model info card
            Card(
              color: Colors.blue[50],
              child: Padding(
                padding: const EdgeInsets.all(12.0),
                child: Row(
                  children: [
                    Icon(
                      Icons.model_training,
                      color: Colors.blue[700],
                      size: 20,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Model:',
                            style: Theme.of(context).textTheme.bodySmall
                                ?.copyWith(
                                  color: Colors.blue[900],
                                  fontWeight: FontWeight.bold,
                                ),
                          ),
                          Text(
                            _modelName,
                            style: Theme.of(context).textTheme.bodyMedium
                                ?.copyWith(color: Colors.blue[900]),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),

            // Status message
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Row(
                  children: [
                    if (_isRunning)
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
                        style: Theme.of(context).textTheme.displayMedium
                            ?.copyWith(
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
                                          fontWeight: isTop
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
                                      fontWeight: isTop
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
                    onPressed:
                        (_isRunning || !_isInitialized || _engine == null)
                        ? null
                        : _startSimulation,
                    icon: Icon(_isRunning ? Icons.stop : Icons.play_arrow),
                    label: Text(_isRunning ? 'Stop' : 'Start Detection'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                  ),
                ),
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
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _stopSimulation();
    _engine?.clear();
    super.dispose();
  }
}
