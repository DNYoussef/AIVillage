/**
 * Model Trainer
 * Trains and fine-tunes ML models for content analysis
 */

import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';

export interface TrainingData {
  text: string;
  labels: {
    toxicity?: number;
    privacy_risk?: number;
    manipulation?: number;
    bias?: number;
    safe?: boolean;
  };
  metadata?: any;
}

export interface TrainingConfig {
  modelType: 'toxicity' | 'privacy' | 'manipulation' | 'bias' | 'combined';
  epochs: number;
  batchSize: number;
  learningRate: number;
  validationSplit: number;
  earlyStopping: boolean;
  patience: number;
  savePath: string;
}

export interface TrainingResult {
  accuracy: number;
  loss: number;
  validationAccuracy: number;
  validationLoss: number;
  epochs: number;
  trainingTime: number;
  modelPath: string;
}

export class ModelTrainer {
  private model: tf.Sequential | null = null;
  private tokenizer: Map<string, number> = new Map();
  private maxSequenceLength: number = 512;
  private vocabularySize: number = 10000;

  constructor(private config: TrainingConfig) {
    this.config = {
      epochs: 10,
      batchSize: 32,
      learningRate: 0.001,
      validationSplit: 0.2,
      earlyStopping: true,
      patience: 3,
      savePath: './models',
      ...config
    };
  }

  /**
   * Train a new model or fine-tune existing one
   */
  async train(data: TrainingData[], existingModelPath?: string): Promise<TrainingResult> {
    const startTime = Date.now();

    console.log(`[Training] Starting ${this.config.modelType} model training...`);
    console.log(`[Training] Dataset size: ${data.length} samples`);

    // Load or create model
    if (existingModelPath) {
      await this.loadModel(existingModelPath);
    } else {
      this.createModel();
    }

    // Prepare data
    const { inputs, labels, validationData } = await this.prepareData(data);

    // Configure callbacks
    const callbacks: tf.CustomCallbackArgs = {
      onEpochEnd: async (epoch, logs) => {
        console.log(`[Training] Epoch ${epoch + 1}/${this.config.epochs}`);
        console.log(`  Loss: ${logs?.loss.toFixed(4)}, Accuracy: ${logs?.acc.toFixed(4)}`);
        console.log(`  Val Loss: ${logs?.val_loss.toFixed(4)}, Val Accuracy: ${logs?.val_acc.toFixed(4)}`);
      }
    };

    // Add early stopping if configured
    const callbackList: tf.CustomCallbackArgs[] = [callbacks];

    if (this.config.earlyStopping) {
      callbackList.push(tf.callbacks.earlyStopping({
        monitor: 'val_loss',
        patience: this.config.patience,
        verbose: 1
      }));
    }

    // Train the model
    const history = await this.model!.fit(inputs, labels, {
      epochs: this.config.epochs,
      batchSize: this.config.batchSize,
      validationData,
      callbacks: callbackList,
      verbose: 0
    });

    // Save the model
    const modelPath = await this.saveModel();

    // Get final metrics
    const finalEpoch = history.history.loss.length - 1;
    const result: TrainingResult = {
      accuracy: history.history.acc[finalEpoch] as number,
      loss: history.history.loss[finalEpoch] as number,
      validationAccuracy: history.history.val_acc[finalEpoch] as number,
      validationLoss: history.history.val_loss[finalEpoch] as number,
      epochs: finalEpoch + 1,
      trainingTime: Date.now() - startTime,
      modelPath
    };

    console.log(`[Training] Training completed in ${result.trainingTime}ms`);
    console.log(`[Training] Final accuracy: ${result.accuracy.toFixed(4)}`);

    // Clean up tensors
    inputs.dispose();
    labels.dispose();
    if (validationData) {
      validationData[0].dispose();
      validationData[1].dispose();
    }

    return result;
  }

  /**
   * Create a new model architecture
   */
  private createModel(): void {
    this.model = tf.sequential();

    switch (this.config.modelType) {
      case 'toxicity':
        this.createToxicityModel();
        break;
      case 'privacy':
        this.createPrivacyModel();
        break;
      case 'manipulation':
        this.createManipulationModel();
        break;
      case 'bias':
        this.createBiasModel();
        break;
      case 'combined':
        this.createCombinedModel();
        break;
    }

    // Compile the model
    this.model.compile({
      optimizer: tf.train.adam(this.config.learningRate),
      loss: this.config.modelType === 'combined' ? 'categoricalCrossentropy' : 'binaryCrossentropy',
      metrics: ['accuracy']
    });

    console.log('[Training] Model architecture created:');
    this.model.summary();
  }

  /**
   * Create toxicity detection model
   */
  private createToxicityModel(): void {
    // Embedding layer
    this.model!.add(tf.layers.embedding({
      inputDim: this.vocabularySize,
      outputDim: 128,
      inputLength: this.maxSequenceLength
    }));

    // LSTM layers
    this.model!.add(tf.layers.lstm({
      units: 128,
      returnSequences: true,
      dropout: 0.2,
      recurrentDropout: 0.2
    }));

    this.model!.add(tf.layers.lstm({
      units: 64,
      dropout: 0.2,
      recurrentDropout: 0.2
    }));

    // Dense layers
    this.model!.add(tf.layers.dense({
      units: 32,
      activation: 'relu'
    }));

    this.model!.add(tf.layers.dropout({ rate: 0.5 }));

    this.model!.add(tf.layers.dense({
      units: 1,
      activation: 'sigmoid'
    }));
  }

  /**
   * Create privacy risk detection model
   */
  private createPrivacyModel(): void {
    // CNN for pattern detection
    this.model!.add(tf.layers.embedding({
      inputDim: this.vocabularySize,
      outputDim: 100,
      inputLength: this.maxSequenceLength
    }));

    this.model!.add(tf.layers.conv1d({
      filters: 128,
      kernelSize: 5,
      activation: 'relu'
    }));

    this.model!.add(tf.layers.globalMaxPooling1d());

    this.model!.add(tf.layers.dense({
      units: 64,
      activation: 'relu'
    }));

    this.model!.add(tf.layers.dropout({ rate: 0.5 }));

    this.model!.add(tf.layers.dense({
      units: 1,
      activation: 'sigmoid'
    }));
  }

  /**
   * Create manipulation detection model
   */
  private createManipulationModel(): void {
    // Bidirectional LSTM for context understanding
    this.model!.add(tf.layers.embedding({
      inputDim: this.vocabularySize,
      outputDim: 128,
      inputLength: this.maxSequenceLength
    }));

    this.model!.add(tf.layers.bidirectional({
      layer: tf.layers.lstm({
        units: 64,
        returnSequences: true
      })
    }));

    this.model!.add(tf.layers.bidirectional({
      layer: tf.layers.lstm({
        units: 32
      })
    }));

    this.model!.add(tf.layers.dense({
      units: 16,
      activation: 'relu'
    }));

    this.model!.add(tf.layers.dropout({ rate: 0.3 }));

    this.model!.add(tf.layers.dense({
      units: 1,
      activation: 'sigmoid'
    }));
  }

  /**
   * Create bias detection model
   */
  private createBiasModel(): void {
    // Attention-based model for bias detection
    this.model!.add(tf.layers.embedding({
      inputDim: this.vocabularySize,
      outputDim: 128,
      inputLength: this.maxSequenceLength
    }));

    // Self-attention layer (simplified)
    this.model!.add(tf.layers.dense({
      units: 128,
      activation: 'tanh'
    }));

    this.model!.add(tf.layers.flatten());

    this.model!.add(tf.layers.dense({
      units: 64,
      activation: 'relu'
    }));

    this.model!.add(tf.layers.dropout({ rate: 0.4 }));

    this.model!.add(tf.layers.dense({
      units: 1,
      activation: 'sigmoid'
    }));
  }

  /**
   * Create combined multi-task model
   */
  private createCombinedModel(): void {
    // Shared layers
    this.model!.add(tf.layers.embedding({
      inputDim: this.vocabularySize,
      outputDim: 256,
      inputLength: this.maxSequenceLength
    }));

    this.model!.add(tf.layers.lstm({
      units: 256,
      returnSequences: true,
      dropout: 0.2
    }));

    this.model!.add(tf.layers.lstm({
      units: 128,
      dropout: 0.2
    }));

    // Task-specific branches would be added here in a functional API
    // For simplicity, using a single output
    this.model!.add(tf.layers.dense({
      units: 64,
      activation: 'relu'
    }));

    this.model!.add(tf.layers.dropout({ rate: 0.5 }));

    // 5 outputs: toxicity, privacy, manipulation, bias, safe
    this.model!.add(tf.layers.dense({
      units: 5,
      activation: 'softmax'
    }));
  }

  /**
   * Prepare training data
   */
  private async prepareData(
    data: TrainingData[]
  ): Promise<{
    inputs: tf.Tensor;
    labels: tf.Tensor;
    validationData?: [tf.Tensor, tf.Tensor];
  }> {
    // Build vocabulary if needed
    if (this.tokenizer.size === 0) {
      this.buildVocabulary(data.map(d => d.text));
    }

    // Tokenize texts
    const sequences = data.map(d => this.tokenize(d.text));

    // Pad sequences
    const paddedSequences = this.padSequences(sequences);

    // Prepare labels based on model type
    let labelData: number[][];

    switch (this.config.modelType) {
      case 'toxicity':
        labelData = data.map(d => [d.labels.toxicity || 0]);
        break;
      case 'privacy':
        labelData = data.map(d => [d.labels.privacy_risk || 0]);
        break;
      case 'manipulation':
        labelData = data.map(d => [d.labels.manipulation || 0]);
        break;
      case 'bias':
        labelData = data.map(d => [d.labels.bias || 0]);
        break;
      case 'combined':
        labelData = data.map(d => [
          d.labels.toxicity || 0,
          d.labels.privacy_risk || 0,
          d.labels.manipulation || 0,
          d.labels.bias || 0,
          d.labels.safe ? 1 : 0
        ]);
        break;
    }

    // Convert to tensors
    const inputs = tf.tensor2d(paddedSequences);
    const labels = tf.tensor2d(labelData);

    // Split validation data
    const splitIndex = Math.floor(data.length * (1 - this.config.validationSplit));

    const trainInputs = inputs.slice([0, 0], [splitIndex, -1]);
    const trainLabels = labels.slice([0, 0], [splitIndex, -1]);

    const valInputs = inputs.slice([splitIndex, 0], [-1, -1]);
    const valLabels = labels.slice([splitIndex, 0], [-1, -1]);

    return {
      inputs: trainInputs,
      labels: trainLabels,
      validationData: [valInputs, valLabels]
    };
  }

  /**
   * Build vocabulary from texts
   */
  private buildVocabulary(texts: string[]): void {
    const wordCounts = new Map<string, number>();

    // Count word frequencies
    for (const text of texts) {
      const words = this.preprocessText(text).split(/\s+/);
      for (const word of words) {
        wordCounts.set(word, (wordCounts.get(word) || 0) + 1);
      }
    }

    // Sort by frequency
    const sortedWords = Array.from(wordCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, this.vocabularySize - 2); // Reserve space for PAD and UNK

    // Build tokenizer
    this.tokenizer.set('<PAD>', 0);
    this.tokenizer.set('<UNK>', 1);

    let index = 2;
    for (const [word] of sortedWords) {
      this.tokenizer.set(word, index++);
    }

    console.log(`[Training] Vocabulary built: ${this.tokenizer.size} tokens`);
  }

  /**
   * Preprocess text
   */
  private preprocessText(text: string): string {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  /**
   * Tokenize text
   */
  private tokenize(text: string): number[] {
    const words = this.preprocessText(text).split(/\s+/);
    return words.map(word =>
      this.tokenizer.get(word) || this.tokenizer.get('<UNK>')!
    );
  }

  /**
   * Pad sequences to fixed length
   */
  private padSequences(sequences: number[][]): number[][] {
    return sequences.map(seq => {
      if (seq.length > this.maxSequenceLength) {
        return seq.slice(0, this.maxSequenceLength);
      } else {
        const padded = new Array(this.maxSequenceLength).fill(0);
        seq.forEach((val, idx) => padded[idx] = val);
        return padded;
      }
    });
  }

  /**
   * Load existing model
   */
  private async loadModel(modelPath: string): Promise<void> {
    this.model = await tf.loadLayersModel(`file://${modelPath}`) as tf.Sequential;
    console.log(`[Training] Loaded existing model from ${modelPath}`);

    // Load tokenizer if it exists
    const tokenizerPath = path.join(path.dirname(modelPath), 'tokenizer.json');
    if (fs.existsSync(tokenizerPath)) {
      const tokenizerData = JSON.parse(fs.readFileSync(tokenizerPath, 'utf-8'));
      this.tokenizer = new Map(tokenizerData);
      console.log(`[Training] Loaded tokenizer with ${this.tokenizer.size} tokens`);
    }
  }

  /**
   * Save model and tokenizer
   */
  private async saveModel(): Promise<string> {
    const timestamp = Date.now();
    const modelDir = path.join(this.config.savePath, `${this.config.modelType}_${timestamp}`);

    // Create directory if it doesn't exist
    if (!fs.existsSync(modelDir)) {
      fs.mkdirSync(modelDir, { recursive: true });
    }

    // Save model
    await this.model!.save(`file://${modelDir}`);
    console.log(`[Training] Model saved to ${modelDir}`);

    // Save tokenizer
    const tokenizerPath = path.join(modelDir, 'tokenizer.json');
    fs.writeFileSync(
      tokenizerPath,
      JSON.stringify(Array.from(this.tokenizer.entries()))
    );
    console.log(`[Training] Tokenizer saved to ${tokenizerPath}`);

    return modelDir;
  }

  /**
   * Evaluate model on test data
   */
  async evaluate(testData: TrainingData[]): Promise<{
    accuracy: number;
    loss: number;
    predictions: number[];
  }> {
    const { inputs, labels } = await this.prepareData(testData);

    const evaluation = await this.model!.evaluate(inputs, labels) as tf.Scalar[];
    const predictions = await (this.model!.predict(inputs) as tf.Tensor).array() as number[][];

    const loss = await evaluation[0].data();
    const accuracy = await evaluation[1].data();

    // Clean up
    inputs.dispose();
    labels.dispose();
    evaluation.forEach(t => t.dispose());

    return {
      loss: loss[0],
      accuracy: accuracy[0],
      predictions: predictions.map(p => p[0])
    };
  }

  /**
   * Generate synthetic training data for testing
   */
  static generateSyntheticData(count: number): TrainingData[] {
    const data: TrainingData[] = [];

    const toxicPhrases = [
      'This is offensive and harmful content',
      'I hate everything about this',
      'You are terrible and worthless'
    ];

    const safePhrases = [
      'This is a helpful and constructive comment',
      'Thank you for your assistance',
      'I appreciate your help with this'
    ];

    const privacyPhrases = [
      'My SSN is 123-45-6789',
      'Call me at 555-0123',
      'My email is user@example.com'
    ];

    for (let i = 0; i < count; i++) {
      const isToxic = Math.random() < 0.3;
      const hasPrivacy = Math.random() < 0.2;
      const hasManipulation = Math.random() < 0.15;
      const hasBias = Math.random() < 0.25;

      let text = '';

      if (isToxic) {
        text = toxicPhrases[Math.floor(Math.random() * toxicPhrases.length)];
      } else if (hasPrivacy) {
        text = privacyPhrases[Math.floor(Math.random() * privacyPhrases.length)];
      } else {
        text = safePhrases[Math.floor(Math.random() * safePhrases.length)];
      }

      data.push({
        text,
        labels: {
          toxicity: isToxic ? Math.random() * 0.5 + 0.5 : Math.random() * 0.3,
          privacy_risk: hasPrivacy ? Math.random() * 0.5 + 0.5 : Math.random() * 0.2,
          manipulation: hasManipulation ? Math.random() * 0.5 + 0.5 : Math.random() * 0.2,
          bias: hasBias ? Math.random() * 0.5 + 0.5 : Math.random() * 0.3,
          safe: !isToxic && !hasPrivacy && !hasManipulation
        }
      });
    }

    return data;
  }

  /**
   * Dispose of model and free resources
   */
  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
  }
}

export default ModelTrainer;