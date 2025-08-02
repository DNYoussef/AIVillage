# Atlantis Dev Onboarding 🚀  (20-minute target)

1. **Clone & install**
   ```bash
   git clone https://github.com/AtlantisAI/atlantis.git && cd atlantis
   poetry install --no-interaction --no-root
   ```
2. **Run checks**
   ```bash
   make lint && make test
   ```
3. **Launch Twin**
   ```bash
   make dev-up   # starts twin container
   http :8000/v1/chat prompt="hello"
   ```
4. **Mobile quick-start** – See [`mobile-app/README.md`](../mobile-app/README.md)

5. **Explore source modules**
   - The `src/` directory now contains **17 modules**, including the new `digital_twin` module for personalized twins.

## Troubleshooting FAQ
| Symptom | Fix |
|---------|-----|
| `mypy` missing stubs | `poetry run mypy --install-types --non-interactive` |
| Docker fails on Apple M-series | `export DOCKER_DEFAULT_PLATFORM=linux/amd64` |
