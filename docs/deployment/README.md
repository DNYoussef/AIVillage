# Deployment Environment Variables

The Docker Compose stack expects sensitive configuration to be supplied via environment variables.
Provide these values in a local `.env` file or through your container orchestrator's secret manager
before starting the services.

Required variables:

- `OPENAI_API_KEY` – API key for OpenAI access
- `NEO4J_URI` – connection URI for the Neo4j graph database
- `NEO4J_USER` – username for the Neo4j graph database
- `NEO4J_PASSWORD` – password for the Neo4j graph database
- `DATABASE_URL` – connection string for the credits ledger database
- `MCP_SERVER_SECRET` – secret used to secure MCP server tokens
- `POSTGRES_PASSWORD` – password for the PostgreSQL credits database
- `REDIS_PASSWORD` – password for the Redis instance
- `GRAFANA_PASSWORD` – admin password for Grafana
- `HYPERAG_JWT_SECRET` – secret used to sign HyperRAG JWT tokens

Example `.env` file:

```dotenv
OPENAI_API_KEY=change-me
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=change-me
DATABASE_URL=postgresql://user:password@localhost/aivillage
MCP_SERVER_SECRET=change-me-change-me-change-me-change-me
POSTGRES_PASSWORD=change-me
REDIS_PASSWORD=change-me
GRAFANA_PASSWORD=change-me
HYPERAG_JWT_SECRET=change-me
```

Ensure the `.env` file is kept out of version control and production secrets are stored in a secure
manager provided by your deployment platform.
