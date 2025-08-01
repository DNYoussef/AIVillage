# AIVillage Directory Structure

## Root Directory
```
AIVillage/
├── .github/                    # GitHub workflows and configurations
├── .vscode/                    # VS Code workspace settings
├── .claude/                    # Claude-specific configurations
├── .mcp/                       # Model Context Protocol configurations
├── .prompts/                   # Prompt templates
├── agent_forge/                # Main agent forge implementation
│   ├── adas/                   # Adaptive Data Augmentation System
│   │   ├── adas.py
│   │   ├── adas_secure.py
│   │   ├── system.py
│   │   ├── tech_pool.py
│   │   └── technique_archive.py
│   ├── bakedquietiot/          # Baked Quiet IoT implementation
│   │   ├── deepbaking.py
│   │   └── quiet_star.py
│   ├── compression/            # Model compression techniques
│   │   ├── stage1.py
│   │   ├── stage1_bitnet.py
│   │   ├── stage1_config.py
│   │   ├── stage2.py
│   │   ├── vptq.py
│   │   ├── seedlm.py
│   │   ├── hyperfn.py
│   │   └── eval_utils.py
│   ├── core/                   # Core functionality
│   │   ├── main.py
│   │   └── model.py
│   ├── deployment/             # Deployment utilities
│   │   └── manifest_generator.py
│   ├── evaluation/             # Evaluation framework
│   │   └── evaluator.py
│   ├── evomerge/               # Evolutionary merging system
│   │   ├── merger.py
│   │   ├── evolutionary_tournament.py
│   │   ├── benchmarks.py
│   │   ├── config.py
│   │   ├── cross_domain.py
│   │   ├── instruction_tuning.py
│   │   └── merging/
│   │       ├── merge_techniques.py
│   │       └── merger.py
│   ├── foundation/             # Foundation models
│   │   ├── bitnet.py
│   │   └── quiet_star.py
│   ├── geometry/               # Geometry-aware components
│   │   ├── id_twonn.py
│   │   └── snapshot.py
│   ├── meta/                   # Meta-learning components
│   │   └── geo2z_policy.py
│   ├── model_compression/      # Model compression utilities
│   │   ├── bitlinearization.py
│   │   └── model_compression.py
│   ├── optim/                  # Optimizers
│   │   ├── augmented_adam.py
│   │   └── grokfast_opt.py
│   ├── phase2/                 # Phase 2 implementations
│   │   ├── pid.py
│   │   └── train_level.py
│   ├── phase3/                 # Phase 3 implementations
│   │   └── self_modeling_gate.py
│   ├── phase4/                 # Phase 4 implementations
│   │   ├── adas.py
│   │   └── prompt_bake.py
│   ├── phase5/                 # Phase 5 implementations
│   │   ├── compress.py
│   │   └── monitor.py
│   ├── prompt_baking/          # Prompt baking system
│   │   ├── baker.py
│   │   ├── loader.py
│   │   └── prompts/
│   │       └── morality_v1.md
│   ├── self_awareness/         # Self-awareness modules
│   │   ├── metacognaitve_eval.py
│   │   ├── self_guided_metacognative_baking.py
│   │   └── text_generation.py
│   ├── sleepdream/             # Sleep and dream learning
│   │   └── sleep_dream.py
│   ├── svf/                    # Singular Value Filtering
│   │   ├── ops.py
│   │   └── svf_ops.py
│   ├── tool_baking/            # Tool baking utilities
│   │   ├── communication_prompt_baker.py
│   │   └── rag_prompt_baker.py
│   ├── training/               # Training pipeline
│   │   ├── curriculum.py
│   │   ├── enhanced_self_modeling.py
│   │   ├── expert_vectors.py
│   │   ├── geometry_pipeline.py
│   │   ├── grokfast.py
│   │   ├── identity.py
│   │   ├── training_loop.py
│   │   └── quiet_star.py
│   └── utils/                  # Utility functions
│       ├── adas.py
│       ├── expert_vector.py
│       ├── grokfast.py
│       ├── hypercomp.py
│       └── vptq.py
├── agents/                     # Agent implementations
│   ├── base/                   # Base agent components
│   │   └── process_handler.py
│   ├── interfaces/             # Agent interfaces
│   │   ├── agent_interface.py
│   │   ├── communication_interface.py
│   │   ├── processing_interface.py
│   │   ├── rag_interface.py
│   │   └── training_interface.py
│   ├── king/                   # KING agent
│   │   ├── analytics/
│   │   ├── input/
│   │   ├── planning/
│   │   ├── task_management/
│   │   ├── king_agent.py
│   │   └── coordinator.py
│   ├── magi/                   # MAGI agent
│   │   └── magi_agent.py
│   ├── sage/                   # SAGE agent
│   │   ├── sage_agent.py
│   │   ├── rag_management.py
│   │   └── knowledge_graph_agent.py
│   └── unified_base_agent.py
├── calibration/                # Calibration system
│   ├── calibrate.py
│   ├── conformal.py
│   └── dataset.py
├── communications/             # Communication protocols
│   ├── protocol.py
│   ├── credits.py
│   ├── credits_api.py
│   ├── mesh_node.py
│   └── mcp_client.py
├── compressed_models/          # Compressed model storage
│   ├── compressed_deepseek-coder-1.3b-base/
│   ├── compressed_qwen1.5-0.5b/
│   └── compressed_tinyllama-1.1b-chat-v0.6/
├── config/                     # Configuration files
├── configs/                    # YAML configurations
│   ├── decision_making.yaml
│   ├── deploy.yaml
│   ├── rag_config.yaml
│   └── training.yaml
├── core/                       # Core system components
│   ├── chat_engine.py
│   ├── communication.py
│   ├── error_handling.py
│   └── logging_config.py
├── data/                       # Data storage
│   ├── agent_data.db
│   └── backups/
├── docs/                       # Documentation
│   ├── architecture.md
│   ├── agent_forge_pipeline_overview.md
│   ├── adr/                    # Architecture Decision Records
│   └── specs/
├── federated/                  # Federated learning
│   └── twin_trainer.py
├── infra/                      # Infrastructure
│   ├── mcp/
│   └── qdrant-dev/
├── ingestion/                  # Data ingestion
│   ├── connectors/
│   └── vector_ds.py
├── logs/                       # Log files
├── mobile-app/                 # Mobile application
│   ├── App.js
│   ├── components/
│   ├── screens/
│   └── services/
├── monitoring/                 # Monitoring configuration
│   ├── prometheus.yml
│   └── grafana/
├── nlp/                        # NLP utilities
│   └── named_entity_recognition.py
├── rag_system/                 # RAG (Retrieval Augmented Generation) system
│   ├── agents/
│   ├── core/
│   ├── error_handling/
│   ├── evaluation/
│   ├── processing/
│   ├── retrieval/
│   ├── tracking/
│   └── utils/
├── schemas/                    # JSON schemas
│   └── evidencepack_v1.json
├── scripts/                    # Utility scripts
│   ├── manage.sh
│   ├── setup_env.sh
│   └── migrate_to_qdrant.py
├── services/                   # Microservices
│   ├── core/
│   ├── gateway/
│   └── twin/
├── tests/                      # Test suite
│   ├── agents/
│   ├── compression/
│   ├── core/
│   └── (various test files)
├── twin_runtime/               # Twin runtime system
│   ├── compressed_loader.py
│   ├── fine_tune.py
│   └── runner.py
├── ui/                         # User interface
│   ├── index.html
│   ├── script.js
│   └── components/
├── vendor/                     # Third-party dependencies
├── main.py                     # Main entry point
├── server.py                   # Server implementation
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Poetry configuration
├── poetry.lock                 # Poetry lock file
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker compose configuration
├── Makefile                    # Build automation
├── README.md                   # Project documentation
├── LICENSE                     # License file
└── (various configuration and documentation files)
```

## Key Directories

### agent_forge/
The core implementation of the Agent Forge system, containing:
- Model compression techniques
- Training pipelines
- Optimization algorithms
- Self-awareness and metacognitive components
- Evolutionary merging system

### agents/
Contains the three main agents:
- **KING**: Knowledge Integration and Navigation Genius
- **SAGE**: Strategic Analysis and Generative Engine
- **MAGI**: Multi-Agent Generative Intelligence

### rag_system/
The Retrieval Augmented Generation system with:
- Knowledge tracking
- Advanced NLP processing
- Hybrid retrieval mechanisms
- Confidence estimation

### communications/
Handles inter-agent communication with:
- Protocol definitions
- Credit management system
- Mesh networking
- MCP (Model Context Protocol) integration

### tests/
Comprehensive test suite covering:
- Unit tests
- Integration tests
- Performance tests
- Soak tests

## Hidden Files and Directories
- `.github/`: GitHub Actions workflows
- `.vscode/`: VS Code settings
- `.claude/`: Claude-specific configurations
- `.mcp/`: MCP server configurations
- `.env`: Environment variables
- `.gitignore`: Git ignore rules
- `.pre-commit-config.yaml`: Pre-commit hooks

## Notable Features
1. Multi-phase training pipeline (phase2-5)
2. Advanced compression techniques (BitNet, VPTQ, SeedLM)
3. Evolutionary model merging
4. Self-modeling and metacognitive capabilities
5. Federated learning support
6. Mobile app integration
7. Comprehensive monitoring with Prometheus/Grafana
8. Docker containerization
9. Credits-based resource management system
