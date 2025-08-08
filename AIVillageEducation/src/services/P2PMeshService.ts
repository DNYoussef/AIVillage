import { NativeModules } from 'react-native';

interface Lesson {
  id: string;
  content: string;
}

interface Peer {
  id: string;
}

export default class P2PMeshService {
  private bridge: any;
  private peers: Map<string, Peer> = new Map();

  async initialize() {
    this.bridge = NativeModules.LibP2PBridge;
    await this.bridge.initialize({
      maxPeers: 50,
      discoveryInterval: 5000,
      messageTimeout: 3000
    });
    this.startDiscovery();
  }

  startDiscovery() {
    this.bridge.onPeerFound((peer: Peer) => {
      this.peers.set(peer.id, peer);
    });
  }

  async findNearbyPeers(): Promise<Peer[]> {
    return Array.from(this.peers.values());
  }

  async compress(lesson: Lesson): Promise<string> {
    return JSON.stringify(lesson); // placeholder
  }

  calculateChecksum(data: string): string {
    let hash = 0;
    for (let i = 0; i < data.length; i++) {
      hash = (hash << 5) - hash + data.charCodeAt(i);
      hash |= 0;
    }
    return hash.toString();
  }

  async shareLesson(lesson: Lesson) {
    const compressed = await this.compress(lesson);
    const peers = await this.findNearbyPeers();
    for (const peer of peers) {
      await this.bridge.sendMessage(peer.id, {
        type: 'LESSON_SHARE',
        payload: compressed,
        checksum: this.calculateChecksum(compressed)
      });
    }
  }
}
