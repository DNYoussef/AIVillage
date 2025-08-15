import { NativeModules } from 'react-native';
import pako from 'pako';
import { createHash } from 'crypto';

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
    const json = JSON.stringify(lesson);
    const compressed = pako.deflate(json);
    return Buffer.from(compressed).toString('base64');
  }

  calculateChecksum(data: string): string {
    return createHash('sha256').update(data, 'base64').digest('hex');
  }

  async shareLesson(lesson: Lesson) {
    const compressed = await this.compress(lesson);
    const checksum = this.calculateChecksum(compressed);
    const peers = await this.findNearbyPeers();
    for (const peer of peers) {
      await this.bridge.sendMessage(peer.id, {
        type: 'LESSON_SHARE',
        payload: {
          data: compressed,
          checksum
        }
      });
    }
  }
}
