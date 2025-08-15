import fs from 'fs';
import path from 'path';

interface DocumentEntry {
  text: string;
  tokens: Set<string>;
}

class HippoCache<V> {
  private cache = new Map<string, { value: V; timestamp: number }>();

  constructor(private maxSize = 100, private ttlMs = 1000 * 60 * 60) {}

  get(key: string): V | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    if (Date.now() - entry.timestamp > this.ttlMs) {
      this.cache.delete(key);
      return null;
    }

    // Mark as recently used
    this.cache.delete(key);
    this.cache.set(key, entry);
    return entry.value;
  }

  set(key: string, value: V): void {
    if (this.cache.size >= this.maxSize) {
      const oldestKey = this.cache.keys().next().value;
      this.cache.delete(oldestKey);
    }
    this.cache.set(key, { value, timestamp: Date.now() });
  }
}

export default class RAGService {
  private documents: DocumentEntry[] = [];
  private cache = new HippoCache<string>();

  constructor() {
    this.loadDocuments();
  }

  private loadDocuments(): void {
    const docsDir = path.resolve(__dirname, '../../../docs');
    if (!fs.existsSync(docsDir)) return;

    const files = fs
      .readdirSync(docsDir)
      .filter((f) => f.endsWith('.md') || f.endsWith('.txt'));

    this.documents = files.map((file) => {
      const text = fs.readFileSync(path.join(docsDir, file), 'utf-8');
      return { text, tokens: new Set(this.tokenize(text)) };
    });
  }

  private tokenize(text: string): string[] {
    return text.toLowerCase().split(/\W+/).filter(Boolean);
  }

  private embed(text: string): Set<string> {
    return new Set(this.tokenize(text));
  }

  private similarity(a: Set<string>, b: Set<string>): number {
    let intersection = 0;
    a.forEach((t) => {
      if (b.has(t)) intersection += 1;
    });
    const union = new Set([...a, ...b]).size;
    return union === 0 ? 0 : intersection / union;
  }

  async answer(query: string): Promise<string> {
    const cached = this.cache.get(query);
    if (cached) return cached;

    const queryTokens = this.embed(query);

    let bestDoc: DocumentEntry | null = null;
    let bestScore = 0;
    for (const doc of this.documents) {
      const score = this.similarity(queryTokens, doc.tokens);
      if (score > bestScore) {
        bestScore = score;
        bestDoc = doc;
      }
    }

    let response: string;
    if (!bestDoc || bestScore === 0) {
      response = `I don't know about "${query}" yet.`;
    } else {
      const sentences = bestDoc.text.split(/(?<=[.!?])\s+/);
      let bestSentence = sentences[0];
      let bestSentenceScore = 0;
      for (const sentence of sentences) {
        const score = this.similarity(queryTokens, this.embed(sentence));
        if (score > bestSentenceScore) {
          bestSentenceScore = score;
          bestSentence = sentence.trim();
        }
      }
      response = bestSentence;
    }

    this.cache.set(query, response);
    return response;
  }
}
