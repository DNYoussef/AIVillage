# AIVillage Clean Architecture & Organization Plan

## Your Vision → Industry Standard Structure

You've identified the perfect logical components. Here's how to organize them following software engineering best practices:

---

## 🎯 CORE PRINCIPLE: Separation of Concerns
Each component should have ONE clear purpose and live in ONE logical location.

---

## RECOMMENDED STRUCTURE

```
AIVillage/
│
├── 📱 apps/                         [User-Facing Applications]
│   ├── mobile/                      
│   │   ├── ios/                    [iOS app]
│   │   ├── android/                [Android app]
│   │   └── shared/                 [Shared mobile code]
│   ├── web/                        [Web interface]
│   └── cli/                        [Command-line tools]
│
├── 🌐 core/                         [Core Business Logic]
│   ├── agents/                     [Meta-Agents System]
│   │   ├── king/                   [King agent]
│   │   ├── magi/                   [Magi agent]
│   │   ├── sage/                   [Sage agent]
│   │   └── common/                 [Shared agent code]
│   │
│   ├── agent-forge/                [Agent Training System]
│   │   ├── training/               [Training pipelines]
│   │   ├── evolution/              [Evolution system]
│   │   ├── curriculum/             [Curriculum learning]
│   │   └── compression/            [Model compression]
│   │
│   ├── hyperrag/                   [RAG System]
│   │   ├── ingestion/              [Data ingestion]
│   │   ├── retrieval/              [Retrieval logic]
│   │   ├── generation/             [Response generation]
│   │   └── cache/                  [Caching layer]
│   │
│   └── tokenomics/                 [DAO & Economics]
│       ├── contracts/              [Smart contracts]
│       ├── governance/             [Voting system]
│       ├── rewards/                [Reward distribution]
│       └── treasury/               [Treasury management]
│
├── 🔗 infrastructure/               [Communication & Networking]
│   ├── p2p/                        [P2P Networks]
│   │   ├── bitchat/                [BitChat protocol]
│   │   ├── betanet/                [BetaNet protocol]
│   │   └── federation/             [Federation layer]
│   │
│   ├── edge/                       [Edge Computing]
│   │   ├── device-management/      [Device registry]
│   │   ├── resource-allocation/    [Resource management]
│   │   └── distributed-compute/    [Computation distribution]
│   │
│   └── api/                        [API Layer]
│       ├── rest/                   [REST APIs]
│       ├── graphql/                [GraphQL APIs]
│       └── websocket/              [Real-time connections]
│
├── 🛠️ devops/                       [Development Operations]
│   ├── ci-cd/                      [CI/CD Pipelines]
│   │   ├── .github/workflows/      [GitHub Actions]
│   │   ├── hooks/                  [Git hooks]
│   │   └── quality-gates/          [Quality checks]
│   │
│   ├── deployment/                 [Deployment Configs]
│   │   ├── docker/                 [Docker files]
│   │   ├── kubernetes/             [K8s manifests]
│   │   └── terraform/              [Infrastructure as Code]
│   │
│   └── monitoring/                 [Monitoring & Logging]
│       ├── metrics/                [Metrics collection]
│       ├── alerts/                 [Alert configs]
│       └── dashboards/             [Monitoring dashboards]
│
├── 🧪 tests/                        [All Testing]
│   ├── unit/                       [Unit tests]
│   ├── integration/                [Integration tests]
│   ├── e2e/                        [End-to-end tests]
│   ├── performance/                [Performance tests]
│   └── fixtures/                   [Test data & mocks]
│
├── 📚 docs/                         [Documentation]
│   ├── architecture/               [Architecture docs]
│   ├── api/                        [API documentation]
│   ├── guides/                     [User guides]
│   ├── development/                [Developer guides]
│   └── deployment/                 [Deployment guides]
│
├── 🔧 scripts/                      [Utility Scripts]
│   ├── setup/                      [Setup scripts]
│   ├── migration/                  [Data migration]
│   ├── maintenance/                [Maintenance scripts]
│   └── analysis/                   [Analysis tools]
│
├── ⚙️ config/                       [Configuration]
│   ├── environments/               [Environment configs]
│   │   ├── development.yaml
│   │   ├── staging.yaml
│   │   └── production.yaml
│   ├── services/                   [Service configs]
│   └── security/                   [Security configs]
│
└── 📦 packages/                    [Shared Libraries]
    ├── common/                     [Common utilities]
    ├── crypto/                     [Cryptography]
    ├── networking/                 [Network utilities]
    └── ml-utils/                   [ML utilities]
```

---

## 🏗️ ARCHITECTURE PRINCIPLES

### 1. **Layered Architecture**
```
┌─────────────────────────────────┐
│         Apps (UI Layer)         │  ← User interfaces
├─────────────────────────────────┤
│      Core (Business Logic)      │  ← Your unique value
├─────────────────────────────────┤
│   Infrastructure (Technical)     │  ← Technical implementation
├─────────────────────────────────┤
│      DevOps (Operations)        │  ← Keep it running
└─────────────────────────────────┘
```

### 2. **Clear Boundaries**
- **Apps**: Only UI and user interaction
- **Core**: Business logic, no infrastructure code
- **Infrastructure**: Technical details, protocols
- **DevOps**: Automation and operations

### 3. **Dependency Rules**
- Dependencies only point DOWNWARD
- Core NEVER depends on Infrastructure
- Apps can use Core and Infrastructure
- Everything can use Packages

---

## 📋 STANDARD FILES AT ROOT

```
AIVillage/
├── README.md                [Project overview]
├── LICENSE                  [Legal]
├── .gitignore              [Git ignore rules]
├── .env.example            [Environment template]
├── package.json            [Node dependencies]
├── requirements.txt        [Python dependencies]
├── Makefile               [Common commands]
├── docker-compose.yml     [Local development]
└── CONTRIBUTING.md        [Contribution guide]
```

---

## 🚀 MIGRATION STRATEGY

### Phase 1: Create New Structure (Day 1-2)
```bash
# Create all directories first
mkdir -p apps/{mobile,web,cli}
mkdir -p core/{agents,agent-forge,hyperrag,tokenomics}
mkdir -p infrastructure/{p2p,edge,api}
mkdir -p devops/{ci-cd,deployment,monitoring}
mkdir -p {tests,docs,scripts,config,packages}
```

### Phase 2: Move Clear Components (Day 3-5)
1. **Move Mobile Apps** → `apps/mobile/`
2. **Move Agents** → `core/agents/`
3. **Move Tests** → `tests/`
4. **Move Docs** → `docs/`

### Phase 3: Consolidate Duplicates (Week 2)
For each duplicate component:
1. Identify best implementation
2. Move to new location
3. Delete redundant versions
4. Update imports

### Phase 4: Clean Up (Week 3)
1. Delete `deprecated/` folder
2. Delete `archive/` folder
3. Remove all backup files
4. Clean temporary directories

---

## 📝 NAMING CONVENTIONS

### Directories
- **lowercase-with-hyphens** for directories
- **Descriptive names** (not abbreviations)
- **Singular for modules** (agent, not agents)
- **Plural for collections** (tests, docs)

### Files
- **snake_case.py** for Python files
- **PascalCase.tsx** for React components
- **camelCase.ts** for TypeScript
- **kebab-case.yaml** for configs

### Code Organization
```python
# Each file should have ONE main class/purpose
# Example: core/agents/king/king_agent.py

class KingAgent:
    """The King meta-agent for coordination."""
    
    def __init__(self):
        # Initialize
        
    def coordinate(self):
        # Main functionality
```

---

## 🎯 YOUR COMPONENTS MAPPED

### 1. **Mobile App** 
- **Location**: `apps/mobile/`
- **Purpose**: User interface for phones
- **Keep separate**: iOS, Android, shared code

### 2. **P2P Communication Layer**
- **Location**: `infrastructure/p2p/`
- **Components**: BitChat, BetaNet, Federation
- **Purpose**: All networking and communication

### 3. **Meta-Agents**
- **Location**: `core/agents/`
- **One folder per agent**: king/, magi/, sage/
- **Shared code**: common/ folder

### 4. **Agent Forge**
- **Location**: `core/agent-forge/`
- **All training**: pipelines, evolution, curriculum
- **Keep together**: All AI training logic

### 5. **HyperRAG System**
- **Location**: `core/hyperrag/`
- **Complete RAG**: ingestion, retrieval, generation
- **Single source of truth**: No duplicates

### 6. **Hardware/Edge**
- **Location**: `infrastructure/edge/`
- **Device management**: Registration, resources
- **Distributed compute**: Task distribution

### 7. **DAO & Tokenomics**
- **Location**: `core/tokenomics/`
- **All economics**: Contracts, voting, rewards
- **Governance**: DAO functionality

### 8. **Automation & CI/CD**
- **Location**: `devops/ci-cd/`
- **GitHub Actions**: workflows/
- **Quality gates**: Linting, testing

### 9. **Tests**
- **Location**: `tests/`
- **All tests**: One central location
- **Organized by type**: unit/, integration/, e2e/

### 10. **Scripts**
- **Location**: `scripts/`
- **Utility scripts**: Setup, migration, maintenance
- **Not core logic**: Only helpers

### 11. **Configs**
- **Location**: `config/`
- **All configuration**: Environment-specific
- **No hardcoded values**: Everything configurable

### 12. **Documentation**
- **Location**: `docs/`
- **All docs**: Architecture, API, guides
- **Keep updated**: Part of development process

---

## ✅ BEST PRACTICES CHECKLIST

### Code Organization
- [ ] One purpose per file
- [ ] Clear module boundaries  
- [ ] No circular dependencies
- [ ] Consistent naming

### Documentation
- [ ] README in each major directory
- [ ] API documentation
- [ ] Architecture diagrams
- [ ] Setup instructions

### Testing
- [ ] Unit tests for business logic
- [ ] Integration tests for APIs
- [ ] E2E tests for critical paths
- [ ] 80% code coverage target

### Configuration
- [ ] Environment variables for secrets
- [ ] Config files for settings
- [ ] No hardcoded values
- [ ] Example configs provided

### Version Control
- [ ] .gitignore properly configured
- [ ] No secrets in repository
- [ ] Clear commit messages
- [ ] Feature branches

### CI/CD
- [ ] Automated testing
- [ ] Code quality checks
- [ ] Security scanning
- [ ] Automated deployment

---

## 🎓 WHY THIS STRUCTURE WORKS

### 1. **Findability**
Anyone can find anything because there's ONE logical place for each thing.

### 2. **Scalability**
New features have clear homes. No confusion about where to add code.

### 3. **Maintainability**
Clear boundaries mean changes don't cascade. Fix one thing without breaking another.

### 4. **Testability**
Separated concerns mean you can test each part independently.

### 5. **Onboarding**
New developers (or AIs) immediately understand the structure.

---

## 🚦 QUICK DECISION GUIDE

**"Where does this code go?"**

1. **Is it UI?** → `apps/`
2. **Is it business logic?** → `core/`
3. **Is it technical infrastructure?** → `infrastructure/`
4. **Is it automation/deployment?** → `devops/`
5. **Is it a test?** → `tests/`
6. **Is it documentation?** → `docs/`
7. **Is it a utility script?** → `scripts/`
8. **Is it configuration?** → `config/`
9. **Is it shared code?** → `packages/`

---

## 📊 SUCCESS METRICS

### Before
- 5,000+ files
- 70% redundancy
- No clear structure
- Hard to find anything
- Duplicates everywhere

### After
- <2,000 files
- 0% redundancy
- Clear structure
- Everything findable
- Single source of truth

---

## 🔄 MAINTENANCE RULES

1. **No code in root directory** (except setup files)
2. **No business logic in infrastructure**
3. **No infrastructure in core**
4. **No production code in tests**
5. **No hardcoded configuration**
6. **Delete dead code immediately**
7. **One implementation per feature**

---

This structure will make your project:
- **Professional** - Industry standard
- **Maintainable** - Easy to update
- **Scalable** - Easy to grow
- **Understandable** - Clear to everyone
- **AI-Friendly** - LLMs can navigate easily

Would you like me to create specific migration scripts to reorganize your codebase into this structure?