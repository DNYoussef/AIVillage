/* eslint-disable @typescript-eslint/no-var-requires */
declare const require: any;

// Attempt to load Node-style modules when available. In browser/React Native
// environments these will simply remain `null` and file based loading will be
// skipped.
const fs: typeof import('fs') | null =
  typeof window === 'undefined' ? (require('fs') as typeof import('fs')) : null;
const path: typeof import('path') | null =
  typeof window === 'undefined' ? (require('path') as typeof import('path')) : null;

interface DocumentEntry {
  id: string;
  text: string;
  embedding: Map<string, number>;
}

interface CacheEntry {
  queryEmbedding: Map<string, number>;
  answer: string;
  timestamp: number;
  accessCount: number;
}

function cosineSim(a: Map<string, number>, b: Map<string, number>): number {
  let dot = 0;
  for (const [k, v] of a) {
    const w = b.get(k);
    if (w) dot += v * w;
  }
  const normA = Math.sqrt(Array.from(a.values()).reduce((s, v) => s + v * v, 0));
  const normB = Math.sqrt(Array.from(b.values()).reduce((s, v) => s + v * v, 0));
  if (normA === 0 || normB === 0) return 0;
  return dot / (normA * normB);
}

class HippoCache {
  private store = new Map<string, CacheEntry>();

  constructor(
    private maxSize = 1000,
    private ttlMs = 24 * 60 * 60 * 1000,
    private similarityThreshold = 0.9,
  ) {}

  get(queryEmbedding: Map<string, number>): CacheEntry | undefined {
    const now = Date.now();
    let bestKey: string | undefined;
    let bestScore = -1;
    for (const [key, entry] of this.store.entries()) {
      if (now - entry.timestamp > this.ttlMs) {
        this.store.delete(key);
        continue;
      }
      const score = cosineSim(queryEmbedding, entry.queryEmbedding);
      if (score > bestScore) {
        bestScore = score;
        bestKey = key;
      }
    }
    if (bestKey && bestScore >= this.similarityThreshold) {
      const entry = this.store.get(bestKey)!;
      entry.accessCount++;
      // maintain LRU order
      this.store.delete(bestKey);
      this.store.set(bestKey, entry);
      return entry;
    }
    return undefined;
  }

  set(key: string, entry: CacheEntry): void {
    if (this.store.size >= this.maxSize) {
      const first = this.store.keys().next().value;
      this.store.delete(first);
    }
    this.store.set(key, entry);
  }
}

export default class RAGService {
  private socket?: WebSocket;
  private cache = new HippoCache();
  private documents: DocumentEntry[] = [];

  private ensureSocket() {
    if (this.socket) return;
    const proto = typeof location !== 'undefined' && location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = typeof location !== 'undefined' ? location.host : 'localhost:8000';
    this.socket = new WebSocket(`${proto}//${host}/ws`);
  }

  private loadDocuments(): void {
    if (this.documents.length || !fs || !path) return;
    const docsDir = path.join(__dirname, '..', 'docs');
    if (!fs.existsSync(docsDir)) return;
    const files = fs
      .readdirSync(docsDir)
      .filter((f) => f.endsWith('.txt') || f.endsWith('.md'));
    for (const file of files) {
      const text = fs.readFileSync(path.join(docsDir, file), 'utf8');
      this.documents.push({ id: file, text, embedding: this.embed(text) });
    }
  }

  private tokenize(text: string): string[] {
    return text.toLowerCase().match(/\b\w+\b/g) ?? [];
  }

  private embed(text: string): Map<string, number> {
    const map = new Map<string, number>();
    for (const token of this.tokenize(text)) {
      map.set(token, (map.get(token) ?? 0) + 1);
    }
    return map;
  }

  async answer(query: string): Promise<string> {
    this.loadDocuments();
    const queryEmbedding = this.embed(query);
    const cached = this.cache.get(queryEmbedding);
    if (cached) return cached.answer;

    let bestDoc: DocumentEntry | undefined;
    let bestScore = 0;
    for (const doc of this.documents) {
      const score = cosineSim(queryEmbedding, doc.embedding);
      if (score > bestScore) {
        bestScore = score;
        bestDoc = doc;
      }
    }
    if (!bestDoc || bestScore < 0.2) {
      return 'No relevant information found.';
    }
    this.cache.set(query, {
      queryEmbedding,
      answer: bestDoc.text,
      timestamp: Date.now(),
      accessCount: 1,
    });
    return bestDoc.text;
  }

  publish(topic: string, data: string): void {
    this.ensureSocket();
    this.socket?.send(JSON.stringify({ topic, data }));
  }

  subscribe(topic: string, handler: (data: string) => void): void {
    this.ensureSocket();
    this.socket?.addEventListener('message', (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.topic === topic) handler(msg.data);
      } catch {
        /* ignore malformed messages */
      }
    });
  }
}
