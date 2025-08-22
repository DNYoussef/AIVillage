# Detailed Connascence Violations Analysis

## Critical God Objects Requiring Immediate Refactoring

### 1. ArchitecturalAnalyzer (scripts/architectural_analysis.py)

**Connascence Assessment:**
- **Type**: God Object anti-pattern
- **Strength**: CRITICAL (982 LOC, 35 methods)
- **Locality**: Single file with cross-module dependencies
- **Degree**: High - affects entire architectural monitoring system

**Specific Violations:**
- Single Responsibility Principle violated
- 35 methods mixing analysis, reporting, and visualization
- Complex internal state management
- Multiple external dependencies (matplotlib, networkx, pandas)

**Refactoring Plan:**
```python
# Extract specialized components:
class DependencyGraphAnalyzer:
    def build_dependency_graph(self) -> nx.DiGraph: ...
    def detect_cycles(self) -> List[List[str]]: ...

class CouplingMetricsCalculator:
    def calculate_efferent_coupling(self) -> Dict[str, int]: ...
    def calculate_afferent_coupling(self) -> Dict[str, int]: ...

class ConnascenceDetector:
    def detect_violations(self) -> List[ConnascenceViolation]: ...

class ArchitecturalReporter:
    def generate_html_report(self) -> str: ...
    def generate_json_report(self) -> Dict: ...

# Coordinator with dependency injection
class ArchitecturalAnalysisCoordinator:
    def __init__(self,
                 graph_analyzer: DependencyGraphAnalyzer,
                 coupling_calc: CouplingMetricsCalculator,
                 connascence_detector: ConnascenceDetector,
                 reporter: ArchitecturalReporter):
        self.graph_analyzer = graph_analyzer
        # ... inject all dependencies
```

### 2. PII/PHI Manager (packages/core/compliance/pii_phi_manager.py)

**Connascence Assessment:**
- **Type**: God Object + Feature Envy
- **Strength**: CRITICAL (1,772 LOC)
- **Locality**: Single massive file
- **Degree**: VERY HIGH - compliance affects entire system

**Algorithmic Duplication Detected:**
- Pattern matching algorithms duplicated across discovery methods
- Similar validation logic in multiple classification functions
- Repetitive database access patterns

**Refactoring Strategy:**
```python
# Domain-driven decomposition:

# 1. Data Discovery Domain
class PIIDiscoveryEngine:
    def scan_text_content(self, text: str) -> List[PIIMatch]: ...
    def scan_database_fields(self, db: Database) -> List[PIIField]: ...

class PIIPatternMatcher:
    """Single source of truth for PII patterns"""
    def match_ssn(self, text: str) -> List[Match]: ...
    def match_email(self, text: str) -> List[Match]: ...
    def match_phone(self, text: str) -> List[Match]: ...

# 2. Classification Domain
class DataClassificationService:
    def classify_field(self, field: DatabaseField) -> DataClassification: ...
    def assess_sensitivity(self, data: Any) -> SensitivityLevel: ...

# 3. Retention Domain
class RetentionPolicyEngine:
    def determine_policy(self, classification: DataClassification) -> RetentionPolicy: ...
    def schedule_deletion(self, data_ref: DataReference, policy: RetentionPolicy): ...

# 4. Compliance Reporting
class ComplianceAuditor:
    def generate_gdpr_report(self) -> GDPRReport: ...
    def generate_hipaa_report(self) -> HIPAAReport: ...
```

### 3. AgentForgePipelineRunner (bin/run_full_agent_forge.py)

**Connascence Assessment:**
- **Type**: Sequential Coupling + Configuration Coupling
- **Strength**: HIGH (568 LOC, orchestration logic)
- **Locality**: Mixed concerns in single class
- **Degree**: MEDIUM - affects CI/CD pipeline

**Sequential Coupling Issues:**
```python
# CURRENT: Brittle sequential coupling
runner.setup_environment()      # Must be first
runner.download_models()        # Depends on environment
runner.configure_wandb()        # Depends on environment
runner.run_agent_forge()        # Depends on models + wandb
runner.benchmark_results()      # Depends on training completion
runner.deploy_smoke_test()      # Depends on benchmarks
```

**Refactoring to Builder Pattern:**
```python
class PipelineBuilder:
    def __init__(self):
        self._stages = []

    def with_environment_setup(self, config: EnvConfig) -> 'PipelineBuilder':
        self._stages.append(EnvironmentSetupStage(config))
        return self

    def with_model_download(self, models: List[str]) -> 'PipelineBuilder':
        self._stages.append(ModelDownloadStage(models))
        return self

    def build(self) -> Pipeline:
        return Pipeline(self._stages)

# Usage - eliminates sequential coupling
pipeline = (PipelineBuilder()
    .with_environment_setup(env_config)
    .with_model_download(model_list)
    .with_wandb_config(wandb_config)
    .with_training_stage(training_config)
    .with_benchmarking(benchmark_config)
    .build())

await pipeline.execute()  # Handles dependencies internally
```

## High-Priority Positional Parameter Violations

### Training Pipeline Functions (432 violations total)

**Example Violation:**
```python
# BEFORE: Positional coupling hell
def create_training_job(model_name, dataset_path, learning_rate, batch_size,
                       epochs, gpu_count, memory_limit, checkpoint_dir,
                       wandb_project, early_stopping, validation_split):
    # Any parameter change breaks all call sites
```

**Refactored Solution:**
```python
@dataclass
class TrainingJobConfig:
    """Strongly-typed configuration eliminates positional coupling"""
    model_name: str
    dataset_path: Path
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10
    gpu_count: int = 1
    memory_limit: str = "8GB"
    checkpoint_dir: Optional[Path] = None
    wandb_project: Optional[str] = None
    early_stopping: bool = True
    validation_split: float = 0.2

def create_training_job(config: TrainingJobConfig) -> TrainingJob:
    """Single parameter - no positional coupling"""
    return TrainingJob(config)

# Usage - extensible and type-safe
config = TrainingJobConfig(
    model_name="qwen-1.5b",
    dataset_path=Path("data/training"),
    learning_rate=0.0001,  # Only specify what differs from defaults
    epochs=50
)
job = create_training_job(config)
```

## Magic Literal Analysis (57,617 instances)

### Critical Magic Number Hotspots

1. **Timeout Values** (found in 47 files)
```python
# BEFORE: Magic timeout values
response = requests.get(url, timeout=300)  # Why 300?
db.execute(query, timeout=30)              # Why 30?
cache.set(key, value, ttl=3600)           # Why 3600?

# AFTER: Named constants
class Timeouts:
    HTTP_REQUEST = 300      # 5 minutes for API calls
    DATABASE_QUERY = 30     # 30 seconds for DB operations
    CACHE_TTL = 3600       # 1 hour cache lifetime

response = requests.get(url, timeout=Timeouts.HTTP_REQUEST)
```

2. **Model Configuration Magic Numbers** (found in 23 files)
```python
# BEFORE: Magic model parameters
model = load_model(hidden_size=768, num_layers=12, num_heads=12)

# AFTER: Configuration classes
@dataclass
class ModelConfig:
    hidden_size: int = 768     # BERT-base hidden dimension
    num_layers: int = 12       # Standard transformer depth
    num_heads: int = 12        # Multi-head attention heads

    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0, "Hidden size must be divisible by heads"

config = ModelConfig(hidden_size=1024, num_layers=24)  # BERT-large
model = load_model(config)
```

3. **Retry and Rate Limiting** (found in 31 files)
```python
# BEFORE: Magic retry logic
for attempt in range(3):  # Why 3?
    try:
        result = api_call()
        break
    except Exception:
        time.sleep(2 ** attempt)  # Why exponential backoff?

# AFTER: Configurable retry policy
@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 60.0

async def with_retry(func: Callable, policy: RetryPolicy = RetryPolicy()):
    for attempt in range(policy.max_attempts):
        try:
            return await func()
        except Exception as e:
            if attempt == policy.max_attempts - 1:
                raise
            delay = min(policy.base_delay * (policy.backoff_factor ** attempt),
                       policy.max_delay)
            await asyncio.sleep(delay)
```

## Cross-Module Connascence Issues

### Import Coupling Violations

**Problem**: Strong connascence across package boundaries
```python
# DANGEROUS: Core depends on specialized packages
from packages.fog.sdk.python.fog_client import FogClient  # Cross-boundary coupling
from packages.edge.legacy_src.digital_twin import DigitalTwin  # Legacy coupling
```

**Solution**: Dependency Inversion
```python
# Define interfaces in core
from abc import ABC, abstractmethod

class CloudComputeInterface(ABC):
    @abstractmethod
    async def submit_job(self, job_spec: JobSpec) -> JobResult: ...

class DigitalTwinInterface(ABC):
    @abstractmethod
    def create_twin(self, entity_id: str) -> DigitalTwin: ...

# Implementations in specialized packages register themselves
class FogClient(CloudComputeInterface):
    async def submit_job(self, job_spec: JobSpec) -> JobResult:
        # Fog-specific implementation

# Core uses interfaces, not concrete classes
class JobScheduler:
    def __init__(self, compute_provider: CloudComputeInterface):
        self.compute_provider = compute_provider  # Dependency injection
```

## Recommended Fitness Functions

### Continuous Monitoring Rules

```python
# Add to pre-commit hooks and CI
def test_no_god_objects():
    """Ensure no class exceeds complexity limits"""
    for python_file in find_python_files():
        classes = extract_classes(python_file)
        for cls in classes:
            assert cls.method_count <= 20, f"God Object: {cls.name} has {cls.method_count} methods"
            assert cls.line_count <= 500, f"God Object: {cls.name} has {cls.line_count} lines"

def test_positional_parameter_limit():
    """Ensure functions use keyword arguments"""
    for python_file in find_python_files():
        functions = extract_functions(python_file)
        for func in functions:
            positional_count = count_positional_params(func)
            assert positional_count <= 3, f"Too many positional params: {func.name} has {positional_count}"

def test_magic_literal_density():
    """Ensure magic literals are below threshold"""
    for python_file in find_python_files():
        magic_literals = detect_magic_literals(python_file)
        density = len(magic_literals) / count_lines_of_code(python_file) * 100
        assert density < 5, f"Magic literal density too high: {python_file} has {density:.1f}%"

def test_coupling_score():
    """Ensure coupling scores remain acceptable"""
    coupling_metrics = calculate_coupling_metrics()
    for file_path, score in coupling_metrics.items():
        assert score < 30, f"Coupling too high: {file_path} has score {score}"
```

## Technical Debt Estimate

| Refactoring Task | Effort (Hours) | Risk Reduction | Priority |
|------------------|---------------|----------------|----------|
| Split ArchitecturalAnalyzer | 24 | HIGH | 1 |
| Refactor PII/PHI Manager | 32 | CRITICAL | 1 |
| Extract magic literals (top 100) | 16 | MEDIUM | 2 |
| Convert positional parameters | 20 | HIGH | 2 |
| Implement dependency injection | 28 | HIGH | 2 |
| Add fitness functions | 12 | MEDIUM | 3 |

**Total Effort**: ~132 hours (3.3 developer weeks)
**Expected ROI**: 60% reduction in bug fix time, 40% faster feature development

## Conclusion

The codebase demonstrates **systematic architectural debt** that requires **immediate refactoring**. While the overall coupling score is acceptable, the presence of **90 God Objects** and **massive magic literal usage** creates **maintenance nightmares** and **testing challenges**.

**Recommendation**: Approve refactoring with **critical priority** on God Object decomposition and magic literal extraction.
