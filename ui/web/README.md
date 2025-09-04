# AIVillage Admin Web UI

React-based administrative dashboard for the Agent Forge pipeline.

## Environment Variables

The dashboard reads API endpoints from environment variables. Defaults point to local services for development.

| Variable | Description | Default |
| --- | --- | --- |
| `VITE_AGENT_FORGE_API_URL` | Agent Forge backend API | `http://localhost:8083` |
| `VITE_AGENT_FORGE_CHAT_API_URL` | Chat/model API | `http://localhost:8084` |
| `VITE_AGENT_FORGE_WS_URL` | WebSocket endpoint | `ws://localhost:8085/ws` |
| `VITE_AGENT_FORGE_AGENT_API_URL` | Agent status API | `http://localhost:8086` |

Create a `.env` file in this directory to override any of these values.

## Development

```bash
npm install
npm run dev
```

