export default class RAGService {
  private socket?: WebSocket;

  private ensureSocket() {
    if (this.socket) return;
    const proto = typeof location !== 'undefined' && location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = typeof location !== 'undefined' ? location.host : 'localhost:8000';
    this.socket = new WebSocket(`${proto}//${host}/ws`);
  }

  async answer(query: string): Promise<string> {
    const res = await fetch('/mcp/hyperrag', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });
    const data = await res.json();
    return data.answer ?? '';
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
