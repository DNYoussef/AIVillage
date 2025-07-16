# Credits Ledger MVP

A fixed-supply shell currency system with 1% burn on spend and Prometheus-based earning for the AIVillage ecosystem.

## Features

- **Fixed Supply**: 1 billion credits maximum supply
- **Burn Mechanism**: 1% burn on all transfers to reduce supply
- **Prometheus Integration**: Automatically mint credits based on node metrics
- **PostgreSQL Backend**: Durable storage with Alembic migrations
- **REST API**: Complete CRUD operations for balances, transfers, and earnings
- **Idempotent Earning**: Prevents double-earning with scrape timestamp keys
- **Docker Integration**: Ready for containerized deployment

## Architecture

### Database Schema

```sql
-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(64) UNIQUE NOT NULL,
    node_id VARCHAR(128) UNIQUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Wallets table
CREATE TABLE wallets (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) UNIQUE,
    balance INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Transactions table
CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    from_user_id INTEGER REFERENCES users(id),
    to_user_id INTEGER REFERENCES users(id),
    amount INTEGER NOT NULL,
    burn_amount INTEGER NOT NULL,
    net_amount INTEGER NOT NULL,
    transaction_type VARCHAR(32) NOT NULL,
    status VARCHAR(32) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- Earnings table (prevents double-earning)
CREATE TABLE earnings (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    scrape_timestamp TIMESTAMP NOT NULL,
    uptime_seconds INTEGER NOT NULL,
    flops INTEGER NOT NULL,
    bandwidth_bytes INTEGER NOT NULL,
    credits_earned INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, scrape_timestamp)
);
```

### Components

1. **Credits Ledger** (`credits_ledger.py`): Core business logic
2. **REST API** (`credits_api.py`): FastAPI endpoints
3. **Worker** (`earn_shells_worker.py`): Prometheus metrics processor
4. **Alembic Migrations**: Database schema management

## API Endpoints

### Authentication
All endpoints except `/health` require authentication (when `API_KEY` is set).

### User Management

#### Create User
```http
POST /users
Content-Type: application/json

{
    "username": "alice",
    "node_id": "node_001"
}
```

**Response:**
```json
{
    "user_id": 1,
    "username": "alice",
    "node_id": "node_001",
    "created_at": "2025-01-16T10:00:00Z",
    "message": "User created successfully"
}
```

### Balance Operations

#### Get Balance
```http
GET /balance/{username}
```

**Response:**
```json
{
    "user_id": 1,
    "username": "alice",
    "balance": 1000,
    "last_updated": "2025-01-16T10:00:00Z"
}
```

#### Transfer Credits
```http
POST /transfer
Content-Type: application/json

{
    "from_username": "alice",
    "to_username": "bob",
    "amount": 100
}
```

**Response:**
```json
{
    "id": 1,
    "from_user": "alice",
    "to_user": "bob",
    "amount": 100,
    "burn_amount": 1,
    "net_amount": 99,
    "transaction_type": "transfer",
    "status": "completed",
    "created_at": "2025-01-16T10:00:00Z",
    "completed_at": "2025-01-16T10:00:01Z"
}
```

### Earning Operations

#### Earn Credits (Prometheus Integration)
```http
POST /earn
Content-Type: application/json

{
    "username": "alice",
    "scrape_timestamp": "2025-01-16T10:00:00Z",
    "uptime_seconds": 3600,
    "flops": 1000000000,
    "bandwidth_bytes": 1000000000
}
```

**Response:**
```json
{
    "id": 1,
    "user_id": 1,
    "scrape_timestamp": "2025-01-16T10:00:00Z",
    "uptime_seconds": 3600,
    "flops": 1000000000,
    "bandwidth_bytes": 1000000000,
    "credits_earned": 1011,
    "created_at": "2025-01-16T10:00:00Z"
}
```

**Earning Formula:**
- Uptime: `(uptime_seconds / 3600) * 10` credits per hour
- FLOPs: `(flops / 1e9) * 1000` credits per GFLOP
- Bandwidth: `(bandwidth_bytes / 1e9) * 1` credits per GB

### Transaction History

#### Get Transactions
```http
GET /transactions/{username}?limit=100
```

**Response:**
```json
[
    {
        "id": 1,
        "from_user": "alice",
        "to_user": "bob",
        "amount": 100,
        "burn_amount": 1,
        "net_amount": 99,
        "transaction_type": "transfer",
        "status": "completed",
        "created_at": "2025-01-16T10:00:00Z",
        "completed_at": "2025-01-16T10:00:01Z"
    }
]
```

### System Information

#### Get Total Supply
```http
GET /supply
```

**Response:**
```json
{
    "total_supply": 999999,
    "max_supply": 1000000000,
    "burn_rate": 0.01,
    "timestamp": "2025-01-16T10:00:00Z"
}
```

#### Health Check
```http
GET /health
```

**Response:**
```json
{
    "status": "healthy",
    "service": "credits-api",
    "timestamp": "2025-01-16T10:00:00Z"
}
```

## Deployment

### Docker Compose

```yaml
services:
  credits-api:
    build:
      context: ./communications
      dockerfile: Dockerfile.credits-api
    ports:
      - "8002:8002"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/credits
      - PROMETHEUS_URL=http://prometheus:9090
    depends_on:
      - postgres

  credits-worker:
    build:
      context: ./communications
      dockerfile: Dockerfile.credits-worker
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/credits
      - PROMETHEUS_URL=http://prometheus:9090
      - CREDITS_API_URL=http://credits-api:8002
      - WORKER_INTERVAL=300
    depends_on:
      - credits-api
      - prometheus

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=credits
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:password@localhost/aivillage` |
| `PROMETHEUS_URL` | Prometheus server URL | `http://localhost:9090` |
| `CREDITS_API_URL` | Credits API URL for worker | `http://localhost:8002` |
| `WORKER_INTERVAL` | Worker interval in seconds | `300` |
| `API_KEY` | Optional API authentication key | None |

### Database Migration

```bash
cd communications
alembic upgrade head
```

## Testing

Run the comprehensive test suite:

```bash
cd communications
python test_credits_standalone.py
```

**Test Coverage:**
- User creation and management
- Balance operations and transfers
- Credit earning with idempotency
- Transaction history
- Error handling and edge cases
- API endpoint functionality

## Worker Operation

The `earn_shells_worker.py` script runs continuously and:

1. Queries Prometheus for active nodes
2. Collects metrics (uptime, FLOPs, bandwidth)
3. Calculates credits earned per node
4. Submits earnings via API (idempotent)
5. Logs all mint events

**Manual Run:**
```bash
python earn_shells_worker.py --prometheus-url http://localhost:9090 --credits-api-url http://localhost:8002
```

**Cron Mode:**
```bash
python earn_shells_worker.py --once
```

## Security Features

- **Input Validation**: Pydantic models with strict validation
- **SQL Injection Protection**: SQLAlchemy ORM with parameterized queries
- **Rate Limiting**: Configurable request limits
- **Idempotent Operations**: Prevents double-spending and double-earning
- **Audit Trail**: Complete transaction history
- **Error Handling**: Comprehensive error responses

## Monitoring

### Prometheus Metrics

The API exports the following metrics:

- `credits_requests_total`: Total API requests by endpoint
- `credits_errors_total`: Total errors by endpoint and type
- `credits_latency_seconds`: Request latency by endpoint

### Logs

- All transactions logged with full details
- Worker mint events logged
- Error conditions logged with context
- Performance metrics tracked

## Example Usage

```bash
# Create users
curl -X POST http://localhost:8002/users \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "node_id": "node_001"}'

# Check balance
curl http://localhost:8002/balance/alice

# Earn credits (simulated)
curl -X POST http://localhost:8002/earn \
  -H "Content-Type: application/json" \
  -d '{
    "username": "alice",
    "scrape_timestamp": "2025-01-16T10:00:00Z",
    "uptime_seconds": 3600,
    "flops": 1000000000,
    "bandwidth_bytes": 1000000000
  }'

# Transfer credits
curl -X POST http://localhost:8002/transfer \
  -H "Content-Type: application/json" \
  -d '{
    "from_username": "alice",
    "to_username": "bob",
    "amount": 100
  }'

# Check transaction history
curl http://localhost:8002/transactions/alice

# Check total supply
curl http://localhost:8002/supply
```

## Development

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set up database
export DATABASE_URL="postgresql://user:pass@localhost/credits"
alembic upgrade head

# Start API server
python credits_api.py

# Start worker (separate terminal)
python earn_shells_worker.py
```

### Adding New Features

1. Update database schema with Alembic migration
2. Add business logic to `credits_ledger.py`
3. Create API endpoints in `credits_api.py`
4. Add comprehensive tests
5. Update documentation

## Troubleshooting

### Common Issues

1. **Database Connection**: Ensure PostgreSQL is running and accessible
2. **Prometheus Metrics**: Check Prometheus URL and node metrics availability
3. **Double Earning**: Worker uses scrape timestamps for idempotency
4. **Insufficient Balance**: Check user balance before transfers
5. **API Authentication**: Set API_KEY environment variable if needed

### Debugging

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python credits_api.py
```

Check database state:
```sql
SELECT u.username, w.balance
FROM users u JOIN wallets w ON u.id = w.user_id;
```

## Future Enhancements

- Multi-currency support
- Advanced earning formulas
- Staking mechanisms
- Cross-chain integration
- Mobile API endpoints
- Real-time notifications
