# Getting Started with AI Village

## Prerequisites

1. Install required software:
   ```bash
   # Python 3.8+
   python -m pip install --upgrade pip
   
   # Redis (for caching)
   # Windows: Download from https://redis.io/download
   # Linux:
   sudo apt-get install redis-server
   
   # PostgreSQL
   # Windows: Download from https://www.postgresql.org/download/
   # Linux:
   sudo apt-get install postgresql
   
   # Node.js (for UI)
   # Download from https://nodejs.org/
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai_village.git
   cd ai_village
   ```

3. Create virtual environment:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   cd ui && npm install && cd ..
   ```

## Configuration

1. Create configuration file:
   ```bash
   cp config/default.yaml config/local.yaml
   ```

2. Edit `config/local.yaml`:
   ```yaml
   environment: development
   database:
     url: postgresql://user:password@localhost:5432/ai_village
   redis:
     url: redis://localhost:6379
   security:
     secret_key: your-secret-key
   ```

3. Set up environment variables:
   ```bash
   # Windows
   set AI_VILLAGE_CONFIG=config/local.yaml
   # Linux/Mac
   export AI_VILLAGE_CONFIG=config/local.yaml
   ```

## Database Setup

1. Create database:
   ```bash
   createdb ai_village
   ```

2. Initialize database:
   ```bash
   python scripts/init_db.py
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

## Testing

1. Run tests:
   ```bash
   pytest tests/
   ```

2. Run specific test suite:
   ```bash
   pytest tests/test_rag_system.py
   pytest tests/test_agents.py
   ```

## Development Workflow

1. Create new branch:
   ```bash
   git checkout -b feature/your-feature
   ```

2. Make changes and test:
   ```bash
   # Run linter
   flake8 .
   
   # Run tests
   pytest
   
   # Run type checking
   mypy .
   ```

3. Submit pull request:
   - Follow contribution guidelines
   - Include tests
   - Update documentation

## Troubleshooting

### Common Issues

1. Database connection:
   ```bash
   # Check PostgreSQL service
   sudo service postgresql status
   
   # Reset database
   dropdb ai_village
   createdb ai_village
   python scripts/init_db.py
   ```

2. Redis connection:
   ```bash
   # Check Redis service
   redis-cli ping
   
   # Clear Redis cache
   redis-cli flushall
   ```

3. API server:
   ```bash
   # Check logs
   tail -f logs/api.log
   
   # Restart server
   kill $(lsof -t -i:8000)
   uvicorn main:app --reload
   ```

### Getting Help

1. Check documentation:
   - `docs/` directory
   - API documentation
   - Code comments

2. Debug logs:
   - `logs/` directory
   - Component-specific logs
   - Error tracking system

## Next Steps

1. Explore the system:
   - Try the chat interface
   - View knowledge graphs
   - Monitor system status

2. Develop new features:
   - Add new agents
   - Enhance RAG system
   - Improve UI

3. Contribute:
   - Read contribution guidelines
   - Submit bug reports
   - Propose enhancements
