# AI Village Education

React Native application for offline-first, voice-driven learning. Features:

- Vosk offline speech recognition
- LibP2P peer-to-peer mesh for lesson sharing
- Digital Twin progress tracking
- Resource aware operation for low-end devices

## Development

```bash
npm install
npx react-native start
npx react-native run-android # or run-ios
```

## Building APK

```bash
cd android
./gradlew assembleRelease
```

## Installation

1. Ensure Android SDK is installed.
2. Clone repository and install dependencies.
3. Build and install APK using commands above.

## Local HyperRAG and MeshNetwork

During development the app can query a locally running HyperRAG MCP server and
exchange messages on the mesh via WebSocket.

### Query HyperRAG

```bash
curl -X POST localhost:8000/mcp/hyperrag \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is HyperRAG?"}'
```

### Mesh WebSocket

```bash
# In one terminal subscribe and send messages
websocat ws://localhost:8000/ws
{"topic":"demo","data":"hello"}
```

These endpoints are only exposed in development and allow the Android service
to mirror Python agent behaviour.
