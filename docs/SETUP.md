# AI Village Setup Guide

## System Requirements

- Python 3.8+
- Redis
- PostgreSQL
- Node.js (for UI)

## Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai_village.git
cd ai_village
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Install UI dependencies:
```bash
cd ui
npm install
cd ..
```

5. Set up database:
```bash
# Create database
createdb ai_village

# Initialize database
python scripts/init_db.py
```

6. Configure environment:
```bash
cp config/default.yaml config/local.yaml
# Edit config/local.yaml with your settings
```

## Running the System

1. Start Redis:
```bash
# Windows: Start Redis service
# Linux:
sudo service redis-server start
```

2. Start API server:
```bash
cd api
uvicorn main:app --reload
```

3. Start UI development server:
```bash
cd ui
npm run dev
```

4. Access the system:
- Dashboard: http://localhost:3000
- API docs: http://localhost:8000/docs

## Component Overview

### 1. RAG System
The RAG (Retrieval-Augmented Generation) system handles knowledge management and retrieval:
- Vector-based retrieval using FAISS
- Graph-based knowledge representation
- Dynamic knowledge updates
- Query optimization

### 2. Agents

#### King Agent
- Task planning and management
- Decision making
- Resource allocation
- Analytics

#### Sage Agent
- Research and knowledge gathering
- RAG system management
- Knowledge synthesis

#### Magi Agent
- Code generation
- Tool creation
- Optimization
- Testing

### 3. Communication System
- Asynchronous message passing
- Priority-based routing
- Real-time updates

### 4. User Interface
- Interactive dashboard
- Knowledge graph visualization
- Decision tree visualization
- Chat interface

## Development Workflow

1. Create feature branch:
```bash
git checkout -b feature/your-feature
```

2. Run tests:
```bash
pytest tests/
```

3. Run linter:
```bash
flake8 .
```

4. Submit pull request:
- Follow contribution guidelines
- Include tests
- Update documentation

## Troubleshooting

### Common Issues

1. Database Connection
```bash
# Check PostgreSQL service
sudo service postgresql status

# Reset database
dropdb ai_village
createdb ai_village
python scripts/init_db.py
```

2. Redis Connection
```bash
# Check Redis service
redis-cli ping

# Clear Redis cache
redis-cli flushall
```

3. API Server
```bash
# Check logs
tail -f logs/api.log

# Restart server
kill $(lsof -t -i:8000)
uvicorn main:app --reload
```

## Monitoring

1. System Health:
- Check dashboard metrics
- Review error logs
- Monitor resource usage

2. Agent Performance:
- View agent metrics
- Check task completion rates
- Review learning progress

3. Knowledge Base:
- Monitor knowledge growth
- Check retrieval performance
- Review integration success

## Security

1. Authentication:
- JWT-based tokens
- Role-based access
- Secure API endpoints

2. Data Protection:
- Encrypted communication
- Secure storage
- Access logging

## Maintenance

1. Regular Tasks:
- Database backup
- Cache cleanup
- Log rotation
- Performance optimization

2. Updates:
- Check for dependency updates
- Apply security patches
- Update documentation

## Support

- GitHub Issues: Report bugs and feature requests
- Documentation: Check latest guides
- Community: Join discussions and get help
