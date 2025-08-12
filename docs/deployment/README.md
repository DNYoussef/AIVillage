# Deployment Environment Variables

The Docker Compose stack expects sensitive configuration to be supplied via environment variables.
Provide these values in a local `.env` file or through your container orchestrator's secret manager
before starting the services.

Required variables:

- `POSTGRES_PASSWORD` – password for the PostgreSQL credits database
- `NEO4J_PASSWORD` – password for the Neo4j graph database
- `REDIS_PASSWORD` – password for the Redis instance
- `GRAFANA_PASSWORD` – admin password for Grafana
- `HYPERAG_JWT_SECRET` – secret used to sign HyperRAG JWT tokens

Example `.env` file:

```dotenv
POSTGRES_PASSWORD=change-me
NEO4J_PASSWORD=change-me
REDIS_PASSWORD=change-me
GRAFANA_PASSWORD=change-me
HYPERAG_JWT_SECRET=change-me
```

Ensure the `.env` file is kept out of version control and production secrets are stored in a secure
manager provided by your deployment platform.
