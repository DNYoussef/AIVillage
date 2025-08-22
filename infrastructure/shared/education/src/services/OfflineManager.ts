import SQLite from 'react-native-sqlite-storage';

interface Lesson {
  id: string;
  gradeLevel: number;
  subject: string;
  audioUrls?: string[];
  images?: string[];
}

class PersistentQueue {
  constructor(private name: string) {}
  private queue: any[] = [];
  isEmpty() { return this.queue.length === 0; }
  async enqueue(item: any) { this.queue.push(item); }
  async dequeue() { return this.queue.shift(); }
}

export default class OfflineManager {
  private db?: SQLite.SQLiteDatabase;
  private syncQueue: PersistentQueue = new PersistentQueue('sync_queue');

  async initialize() {
    this.db = await SQLite.openDatabase({ name: 'aivillage.db' });
    await this.db.executeSql(`
      CREATE TABLE IF NOT EXISTS lessons (
        id TEXT PRIMARY KEY,
        content TEXT,
        grade_level INTEGER,
        subject TEXT,
        cached_at INTEGER
      );
    `);
    await this.db.executeSql(`
      CREATE TABLE IF NOT EXISTS progress (
        lesson_id TEXT,
        completed_at INTEGER,
        score REAL,
        time_spent INTEGER
      );
    `);
  }

  async cacheLesson(lesson: Lesson) {
    await this.db?.executeSql(
      'INSERT OR REPLACE INTO lessons VALUES (?, ?, ?, ?, ?)',
      [lesson.id, JSON.stringify(lesson), lesson.gradeLevel, lesson.subject, Date.now()]
    );
  }

  async hasConnection(): Promise<boolean> {
    // Check network connectivity status
    try {
      const response = await fetch('https://httpbin.org/status/200', {
        method: 'HEAD',
        cache: 'no-cache',
        timeout: 5000
      } as any);
      return response.ok;
    } catch {
      return false;
    }
  }

  async syncItem(item: any) {
    // Sync item with remote server
    try {
      if (item.type === 'progress') {
        // Sync lesson progress to remote API
        console.log('Syncing progress item:', item.id);
        // await api.syncProgress(item);
      } else if (item.type === 'lesson') {
        // Sync lesson data to remote API
        console.log('Syncing lesson item:', item.id);
        // await api.syncLesson(item);
      }
    } catch (error) {
      console.error('Sync failed for item:', item.id, error);
      throw error;
    }
  }

  async syncWhenOnline() {
    if (await this.hasConnection()) {
      while (!this.syncQueue.isEmpty()) {
        const item = await this.syncQueue.dequeue();
        await this.syncItem(item);
      }
    }
  }
}
