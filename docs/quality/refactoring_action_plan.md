# Refactoring Action Plan: Connascence-Based Architecture Improvement

## Sprint Planning for Architectural Debt Reduction

### Phase 1: Critical God Object Decomposition (Week 1-2)

#### Sprint 1.1: ArchitecturalAnalyzer Refactoring (5 days)

**Day 1: Extract Analysis Components**
```python
# Create specialized analyzers
class DependencyGraphAnalyzer:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.graph = nx.DiGraph()
    
    def build_dependency_graph(self) -> nx.DiGraph:
        """Single responsibility: dependency graph construction"""
        # Extract existing logic from ArchitecturalAnalyzer
        pass
    
    def detect_circular_dependencies(self) -> List[List[str]]:
        """Single responsibility: cycle detection"""
        pass

class CouplingMetricsCalculator:
    def calculate_file_coupling(self, file_path: Path) -> CouplingMetric:
        """Single responsibility: coupling calculation"""
        pass
```

**Day 2: Extract Reporting Components**
```python
class ArchitecturalReporter:
    def __init__(self, template_engine: TemplateEngine):
        self.template_engine = template_engine
    
    def generate_html_report(self, metrics: ArchitecturalMetrics) -> str:
        """Single responsibility: HTML report generation"""
        pass
    
    def generate_json_report(self, metrics: ArchitecturalMetrics) -> Dict:
        """Single responsibility: JSON report generation"""
        pass
```

**Day 3: Create Coordinator with Dependency Injection**
```python
class ArchitecturalAnalysisCoordinator:
    def __init__(self, 
                 dependency_analyzer: DependencyGraphAnalyzer,
                 coupling_calculator: CouplingMetricsCalculator,
                 reporter: ArchitecturalReporter):
        self.dependency_analyzer = dependency_analyzer
        self.coupling_calculator = coupling_calculator
        self.reporter = reporter
    
    def perform_full_analysis(self) -> ArchitecturalReport:
        """Orchestrate analysis without doing the work itself"""
        graph = self.dependency_analyzer.build_dependency_graph()
        coupling = self.coupling_calculator.calculate_coupling_metrics(graph)
        return self.reporter.create_report(graph, coupling)
```

**Day 4: Update Integration Points**
- Update scripts/architectural_analysis.py to use new coordinator
- Update CI integration to use new components
- Maintain backward compatibility

**Day 5: Testing & Validation**
- Unit tests for each extracted component
- Integration tests for coordinator
- Performance validation (should be faster due to better separation)

#### Sprint 1.2: PII/PHI Manager Decomposition (5 days)

**Day 1: Domain Analysis & Interface Design**
```python
# Define clean interfaces for each domain
class PIIDiscoveryInterface(ABC):
    @abstractmethod
    def scan_content(self, content: str) -> List[PIIMatch]: ...

class ClassificationInterface(ABC):
    @abstractmethod
    def classify_data(self, data: Any) -> DataClassification: ...

class RetentionInterface(ABC):
    @abstractmethod
    def determine_policy(self, classification: DataClassification) -> RetentionPolicy: ...
```

**Day 2: Extract Discovery Engine**
```python
class PIIDiscoveryEngine:
    def __init__(self, pattern_matcher: PIIPatternMatcher):
        self.pattern_matcher = pattern_matcher  # Inject dependency
    
    def scan_text_content(self, text: str) -> List[PIIMatch]:
        """Focus only on PII discovery in text"""
        matches = []
        matches.extend(self.pattern_matcher.find_ssn_patterns(text))
        matches.extend(self.pattern_matcher.find_email_patterns(text))
        matches.extend(self.pattern_matcher.find_phone_patterns(text))
        return matches

class PIIPatternMatcher:
    """Single source of truth for PII regex patterns"""
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    
    def find_ssn_patterns(self, text: str) -> List[PIIMatch]:
        # Extracted from original huge class
        pass
```

**Day 3: Extract Classification Service**
```python
class DataClassificationService:
    def __init__(self, classification_rules: ClassificationRules):
        self.rules = classification_rules
    
    def classify_field(self, field: DatabaseField) -> DataClassification:
        """Focus only on data classification logic"""
        if self.rules.is_health_data(field):
            return DataClassification.PHI
        elif self.rules.is_personal_data(field):
            return DataClassification.PII
        # ... rest of classification logic
```

**Day 4: Extract Retention Policy Engine**
```python
class RetentionPolicyEngine:
    def __init__(self, policy_config: RetentionConfig):
        self.config = policy_config
    
    def determine_policy(self, classification: DataClassification) -> RetentionPolicy:
        """Focus only on retention policy determination"""
        return self.config.get_policy_for_classification(classification)
    
    def schedule_deletion(self, data_ref: DataReference, policy: RetentionPolicy):
        """Focus only on deletion scheduling"""
        deletion_date = datetime.now() + policy.retention_period
        self.deletion_scheduler.schedule(data_ref, deletion_date)
```

**Day 5: Integration & Testing**
- Create coordinator for PII/PHI management
- Update all integration points
- Comprehensive testing of extracted components

### Phase 2: Magic Literal Extraction (Week 3)

#### Priority Magic Literal Categories

**Day 1-2: Configuration Constants**
```python
# Create configuration modules
class APIConfig:
    DEFAULT_TIMEOUT = 300  # 5 minutes
    RETRY_ATTEMPTS = 3
    RATE_LIMIT_PER_MINUTE = 60
    MAX_PAYLOAD_SIZE = 10 * 1024 * 1024  # 10MB

class ModelConfig:
    BERT_BASE_HIDDEN_SIZE = 768
    BERT_BASE_LAYERS = 12
    BERT_BASE_ATTENTION_HEADS = 12
    
    BERT_LARGE_HIDDEN_SIZE = 1024
    BERT_LARGE_LAYERS = 24
    BERT_LARGE_ATTENTION_HEADS = 16

class DatabaseConfig:
    CONNECTION_POOL_SIZE = 20
    QUERY_TIMEOUT = 30  # seconds
    RETRY_DELAY = 5     # seconds
```

**Day 3-4: Replace Magic Literals in Top 20 Files**
```python
# Before: Magic numbers scattered everywhere
def process_batch(items):
    batch_size = 32  # Magic number
    timeout = 300    # Magic number
    max_retries = 3  # Magic number

# After: Use configuration constants
def process_batch(items, config: ProcessingConfig = ProcessingConfig()):
    batch_size = config.BATCH_SIZE
    timeout = config.TIMEOUT
    max_retries = config.MAX_RETRIES
```

**Day 5: Validation & Testing**
- Ensure all magic literal replacements work correctly
- Add tests to prevent magic literal regression
- Update documentation

### Phase 3: Positional Parameter Refactoring (Week 4)

#### Configuration Object Pattern Implementation

**Day 1-2: Training Pipeline Configuration**
```python
@dataclass
class TrainingConfig:
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
    
    def __post_init__(self):
        """Validate configuration on creation"""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if not 0 < self.validation_split < 1:
            raise ValueError("Validation split must be between 0 and 1")

# Convert from positional to configuration-based
def create_training_job(config: TrainingConfig) -> TrainingJob:
    return TrainingJob(config)
```

**Day 3-4: API and Service Configuration**
```python
@dataclass  
class APIEndpointConfig:
    url: str
    timeout: int = APIConfig.DEFAULT_TIMEOUT
    retries: int = APIConfig.RETRY_ATTEMPTS
    rate_limit: int = APIConfig.RATE_LIMIT_PER_MINUTE
    auth_token: Optional[str] = None
    
@dataclass
class DatabaseConnectionConfig:
    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int = DatabaseConfig.CONNECTION_POOL_SIZE
    timeout: int = DatabaseConfig.QUERY_TIMEOUT
```

**Day 5: Integration & Testing**
- Update all function calls to use configuration objects
- Ensure backward compatibility where needed
- Add comprehensive tests

### Phase 4: Dependency Injection Implementation (Week 5)

#### Core Service Interfaces

**Day 1-2: Define Service Interfaces**
```python
class DatabaseInterface(ABC):
    @abstractmethod
    async def execute_query(self, query: str, params: Dict) -> QueryResult: ...
    
    @abstractmethod
    async def begin_transaction(self) -> TransactionContext: ...

class CacheInterface(ABC):
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]: ...
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int) -> None: ...

class MessageQueueInterface(ABC):
    @abstractmethod
    async def publish(self, topic: str, message: Dict) -> None: ...
    
    @abstractmethod
    async def subscribe(self, topic: str, handler: Callable) -> None: ...
```

**Day 3-4: Implement Dependency Injection Container**
```python
class DIContainer:
    def __init__(self):
        self._services = {}
        self._singletons = {}
    
    def register_singleton(self, interface: Type, implementation: Type):
        """Register a singleton service"""
        self._services[interface] = (implementation, 'singleton')
    
    def register_transient(self, interface: Type, implementation: Type):
        """Register a transient service"""
        self._services[interface] = (implementation, 'transient')
    
    def get(self, interface: Type) -> Any:
        """Resolve service instance"""
        if interface not in self._services:
            raise ValueError(f"Service {interface} not registered")
        
        implementation, lifecycle = self._services[interface]
        
        if lifecycle == 'singleton':
            if interface not in self._singletons:
                self._singletons[interface] = self._create_instance(implementation)
            return self._singletons[interface]
        else:
            return self._create_instance(implementation)
```

**Day 5: Update Core Services**
```python
class UserService:
    def __init__(self, 
                 database: DatabaseInterface,
                 cache: CacheInterface,
                 message_queue: MessageQueueInterface):
        self.database = database
        self.cache = cache
        self.message_queue = message_queue
    
    async def create_user(self, user_data: CreateUserRequest) -> User:
        # No global dependencies - everything injected
        async with self.database.begin_transaction() as tx:
            user = await tx.execute_query("INSERT INTO users...", user_data.dict())
            await self.cache.set(f"user:{user.id}", user, ttl=3600)
            await self.message_queue.publish("user.created", {"user_id": user.id})
            return user
```

### Phase 5: Architectural Fitness Functions (Week 6)

#### Continuous Architecture Validation

**Day 1-2: Implement Fitness Functions**
```python
class ArchitecturalFitnessTests:
    def test_no_god_objects(self):
        """Prevent God Objects from being introduced"""
        violations = []
        for python_file in self.find_python_files():
            classes = self.extract_classes(python_file)
            for cls in classes:
                if cls.method_count > 20:
                    violations.append(f"God Object: {cls.name} has {cls.method_count} methods")
                if cls.line_count > 500:
                    violations.append(f"God Object: {cls.name} has {cls.line_count} lines")
        
        assert not violations, f"God Objects detected: {violations}"
    
    def test_coupling_thresholds(self):
        """Ensure coupling doesn't exceed thresholds"""
        coupling_metrics = self.calculate_coupling_metrics()
        violations = []
        
        for file_path, metrics in coupling_metrics.items():
            if metrics.coupling_score > 30:
                violations.append(f"High coupling: {file_path} score={metrics.coupling_score}")
        
        assert not violations, f"Coupling violations: {violations}"
    
    def test_positional_parameter_limits(self):
        """Prevent functions with too many positional parameters"""
        violations = []
        for python_file in self.find_python_files():
            functions = self.extract_functions(python_file)
            for func in functions:
                pos_params = self.count_positional_params(func)
                if pos_params > 3:
                    violations.append(f"Too many positional params: {func.name} has {pos_params}")
        
        assert not violations, f"Positional parameter violations: {violations}"
```

**Day 3-4: CI Integration**
```yaml
# .github/workflows/architecture_fitness.yml
name: Architecture Fitness Tests
on: [push, pull_request]

jobs:
  architecture_fitness:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
          pip install -e .
      
      - name: Run Architecture Fitness Tests
        run: |
          python -m pytest tests/architecture/ -v
          python scripts/check_connascence.py . --severity high --format json > connascence_report.json
          python scripts/coupling_metrics.py . --format json > coupling_report.json
      
      - name: Upload Architecture Reports
        uses: actions/upload-artifact@v3
        with:
          name: architecture-reports
          path: |
            connascence_report.json
            coupling_report.json
```

**Day 5: Monitoring Dashboard Setup**
```python
class ArchitecturalMetricsDashboard:
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
    
    def generate_dashboard(self) -> str:
        """Generate HTML dashboard with current metrics"""
        metrics = self.metrics_collector.collect_current_metrics()
        
        dashboard_data = {
            'god_object_count': len(metrics.god_objects),
            'average_coupling_score': metrics.average_coupling_score,
            'magic_literal_density': metrics.magic_literal_density,
            'positional_param_violations': len(metrics.positional_violations),
            'trend_data': metrics.historical_trends
        }
        
        return self.render_dashboard_template(dashboard_data)
```

## Success Metrics & Validation

### Baseline Measurements (Current State)
- God Objects: 90
- Average Coupling Score: 15.9/100
- Magic Literal Density: 38.26/100 LOC
- Positional Parameter Violations: 432
- Files >500 LOC: 13

### Target Measurements (Post-Refactoring)
- God Objects: <10 (89% reduction)
- Average Coupling Score: <20 (maintain)
- Magic Literal Density: <5/100 LOC (87% reduction)
- Positional Parameter Violations: <50 (88% reduction)
- Files >500 LOC: 0 (100% reduction)

### Weekly Progress Tracking
```python
# Track weekly progress
Week 1: God Objects reduced from 90 → 60 (33% reduction)
Week 2: God Objects reduced from 60 → 30 (50% reduction) 
Week 3: Magic Literals density from 38.26% → 20%
Week 4: Positional params from 432 → 200
Week 5: Coupling maintained while adding DI
Week 6: All fitness functions passing in CI
```

## Risk Mitigation

### Rollback Strategy
- Maintain feature flags for new components
- Keep original implementations during transition
- Comprehensive regression testing
- Gradual migration with backward compatibility

### Team Communication
- Daily standups during critical refactoring weeks
- Code review focus on connascence principles
- Pair programming for complex extractions
- Documentation updates in parallel

## Expected Outcomes

### Technical Benefits
- 60% reduction in bug fix time
- 40% faster feature development  
- 80% improvement in test execution speed
- 90% reduction in coupling-related incidents

### Maintainability Benefits
- Clearer separation of concerns
- Easier unit testing
- Reduced cognitive load
- Better onboarding for new developers

### Quality Benefits
- Architectural fitness functions prevent regression
- Continuous monitoring of coupling metrics
- Automated connascence violation detection
- Measurable technical debt reduction