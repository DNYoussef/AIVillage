# AIVillage Installation Guide

This comprehensive guide covers all installation methods and requirements for setting up AIVillage.

## 🎯 Quick Start (Recommended)

### Prerequisites
- Python 3.9+ (Python 3.11+ recommended)
- Git with LFS support
- 4GB+ available RAM
- 10GB+ available disk space

### 1. Clone Repository
```bash
git clone https://github.com/DNYoussef/AIVillage.git
cd AIVillage
```

### 2. Install Dependencies

#### Option A: Full Installation (Recommended for Production)
```bash
# Install core dependencies with constraints for stability
pip install -r config/requirements/requirements.txt -c config/constraints.txt
```

#### Option B: Development Installation
```bash
# Install development dependencies (includes testing, linting tools)
pip install -r config/requirements/requirements-dev.txt -c config/constraints.txt
```

#### Option C: Minimal Installation
```bash
# Install only main runtime dependencies
pip install -r config/requirements/requirements-main.txt -c config/constraints.txt
```

### 3. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your configuration
# At minimum, set JWT_SECRET and SECRET_KEY
```

### 4. Deploy System
```bash
# Deploy the enhanced fog computing system
python scripts/deploy_enhanced_fog_system.py

# Verify all systems operational
python scripts/test_enhanced_fog_integration.py
```

### 5. Start Services
```bash
# Start the main API gateway
cd infrastructure/gateway
python enhanced_unified_api_gateway.py
```

Access the system:
- **API Gateway**: http://localhost:8000
- **Admin Dashboard**: http://localhost:8000/admin_interface.html
- **API Documentation**: http://localhost:8000/docs

## 📦 Installation Options

### Standard Installation
For most users who want to run AIVillage:
```bash
pip install -r config/requirements/requirements.txt -c config/constraints.txt
```

### Development Installation
For contributors and developers:
```bash
pip install -r config/requirements/requirements-dev.txt -c config/constraints.txt
```

### Production Installation
For production deployments:
```bash
pip install -r config/requirements/requirements-production.txt -c config/constraints.txt
```

### Security-Enhanced Installation
For security-focused deployments:
```bash
pip install -r config/requirements/requirements-security.txt -c config/constraints.txt
```

### Experimental Features
For testing cutting-edge features:
```bash
pip install -r config/requirements/requirements-experimental.txt -c config/constraints.txt
```

## 🔧 Advanced Setup

### Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r config/requirements/requirements.txt -c config/constraints.txt
```

### Using Conda
```bash
# Create conda environment
conda create -n aivillage python=3.11
conda activate aivillage

# Install dependencies
pip install -r config/requirements/requirements.txt -c config/constraints.txt
```

### Docker Setup (Alternative)
```bash
# Build Docker image
docker build -t aivillage .

# Run container
docker run -p 8000:8000 aivillage
```

## 📋 Requirements Files Overview

- **`requirements.txt`** - Main runtime dependencies
- **`requirements-main.txt`** - Minimal core dependencies
- **`requirements-dev.txt`** - Development and testing tools
- **`requirements-production.txt`** - Production-optimized packages
- **`requirements-security.txt`** - Security-enhanced packages
- **`requirements-experimental.txt`** - Experimental features
- **`requirements-test.txt`** - Testing frameworks only
- **`constraints.txt`** - Version constraints for stability

## 🛠️ Component-Specific Installation

### For P2P Features
```bash
pip install -r infrastructure/p2p/communications/requirements.txt
```

### For Gateway Services
```bash
pip install -r infrastructure/gateway/requirements.txt
```

### For Agent Forge Training
```bash
pip install -r core/agent-forge/benchmarks/requirements.txt
```

## 🚨 Troubleshooting

### Common Issues

#### Issue: `neo4j` installation fails
**Solution**: Install Microsoft C++ Build Tools or Visual Studio Build Tools
```bash
# Alternative: Use conda
conda install neo4j-python-driver
```

#### Issue: `faiss-cpu` installation fails
**Solution**: Try with conda or use alternative package
```bash
conda install faiss-cpu -c conda-forge
# Or for CUDA systems:
pip install faiss-gpu
```

#### Issue: Permission denied errors
**Solution**: Use user installation
```bash
pip install --user -r config/requirements/requirements.txt -c config/constraints.txt
```

#### Issue: Dependency conflicts
**Solution**: Use constraints file and clean environment
```bash
pip uninstall -r config/requirements/requirements.txt -y
pip install -r config/requirements/requirements.txt -c config/constraints.txt
```

### Verification Commands

#### Test Installation
```bash
# Quick system test
python scripts/test_enhanced_fog_integration.py

# Validate dependencies
pip check

# Security audit
pip-audit

# Run basic tests
pytest tests/ -v --tb=short -x
```

#### Code Quality Checks
```bash
# Format code
black . --check --diff

# Lint code
ruff check . --fix

# Type checking
mypy . --ignore-missing-imports
```

## 🔒 Security Considerations

### Required Environment Variables
Set these in your `.env` file:
```env
JWT_SECRET=your_super_secure_jwt_secret_key_at_least_32_characters_long
SECRET_KEY=your_super_secure_app_secret_key_at_least_32_characters_long
```

### Generate Secure Keys
```bash
# Generate JWT secret
python -c "import secrets; print('JWT_SECRET=' + secrets.token_hex(32))"

# Generate app secret
python -c "import secrets; print('SECRET_KEY=' + secrets.token_hex(32))"
```

## 📊 System Requirements

### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 5GB
- **Python**: 3.9+

### Recommended Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 20GB+
- **Python**: 3.11+

### Production Requirements
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 50GB+ (SSD recommended)
- **Python**: 3.11+
- **OS**: Ubuntu 22.04+, Windows 11, macOS 13+

## 🔄 Update Process

### Update Dependencies
```bash
# Update to latest versions
pip install --upgrade -r config/requirements/requirements.txt -c config/constraints.txt

# Update constraints (review carefully)
pip-compile config/requirements/requirements.txt --upgrade
```

### Update System
```bash
git pull origin main
pip install --upgrade -r config/requirements/requirements.txt -c config/constraints.txt
python scripts/deploy_enhanced_fog_system.py
```

## 💡 Performance Optimization

### For Development
```bash
# Install development dependencies for faster testing
pip install -r config/requirements/requirements-dev.txt -c config/constraints.txt
```

### For Production
```bash
# Use production-optimized packages
pip install -r config/requirements/requirements-production.txt -c config/constraints.txt
```

## 📞 Getting Help

If you encounter issues:
1. Check [Common Issues](#common-issues) above
2. Review [GitHub Issues](https://github.com/DNYoussef/AIVillage/issues)
3. Check the [Architecture Documentation](../architecture/ARCHITECTURE.md)
4. Review the [Contributing Guide](../../CONTRIBUTING.md)

## ✅ Verification Checklist

After installation, verify:
- [ ] All dependencies installed successfully
- [ ] `.env` file configured with secure keys
- [ ] System deployment completes without errors
- [ ] API gateway starts on port 8000
- [ ] Admin dashboard loads at `/admin_interface.html`
- [ ] API documentation accessible at `/docs`
- [ ] Basic tests pass with `pytest`

## 🎉 Next Steps

After successful installation:
1. Review the [Architecture Overview](../architecture/ARCHITECTURE.md)
2. Check out the [API Documentation](http://localhost:8000/docs)
3. Explore the [Admin Dashboard](http://localhost:8000/admin_interface.html)
4. Read the [Contributing Guide](../../CONTRIBUTING.md) to get involved

---

**Last Updated**: August 29, 2025  
**Version**: 2.1.0  
**Tested Platforms**: Windows 11, Ubuntu 22.04, macOS 13+