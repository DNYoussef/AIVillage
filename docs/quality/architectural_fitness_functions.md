# Architectural Fitness Functions
## Continuous Architecture Validation for AIVillage

---

## 🎯 Purpose

Architectural fitness functions are automated checks that verify architectural characteristics remain within acceptable bounds during development. These functions prevent architectural drift and ensure continuous compliance with design principles.

---

## 📏 Connascence-Based Fitness Functions

### 1. God Object Prevention
```python
def test_no_god_objects():
    """Prevent classes exceeding size thresholds"""
    for file_path in get_python_files():
        classes = extract_classes(file_path)
        for class_info in classes:
            assert class_info.line_count <= 500, f"God Object detected: {class_info.name} ({class_info.line_count} lines)"
            assert class_info.method_count <= 20, f"Too many methods: {class_info.name} ({class_info.method_count} methods)"
```

### 2. Strong Connascence Locality
```python
def test_strong_connascence_locality():
    """Ensure strong connascence stays within module boundaries"""
    violations = check_cross_module_connascence()
    assert len(violations) == 0, f"Cross-module strong connascence detected: {violations}"
```

### 3. Magic Literal Threshold
```python
def test_magic_literal_density():
    """Limit magic literals in business logic"""
    for file_path in get_business_logic_files():
        density = calculate_magic_literal_density(file_path)
        assert density <= 10, f"Excessive magic literals in {file_path}: {density}%"
```

### 4. Positional Parameter Limit
```python
def test_positional_parameter_limit():
    """Enforce keyword-only parameters for complex functions"""
    for file_path in get_python_files():
        functions = extract_functions(file_path)
        for func in functions:
            if len(func.positional_params) > 3:
                assert func.has_keyword_only, f"Function {func.name} needs keyword-only parameters"
```

---

## 🏗️ Structural Fitness Functions

### 5. Circular Dependency Detection
```python
def test_no_circular_dependencies():
    """Prevent circular imports between packages"""
    dependency_graph = build_dependency_graph()
    cycles = detect_cycles(dependency_graph)
    assert len(cycles) == 0, f"Circular dependencies detected: {cycles}"
```

### 6. Module Coupling Limits
```python
def test_module_coupling_limits():
    """Ensure modules don't exceed coupling thresholds"""
    for module in get_modules():
        afferent_coupling = calculate_afferent_coupling(module)
        efferent_coupling = calculate_efferent_coupling(module)
        instability = efferent_coupling / (afferent_coupling + efferent_coupling)

        assert instability <= 0.8, f"Module {module} too unstable: {instability}"
        assert afferent_coupling <= 10, f"Module {module} too coupled: {afferent_coupling} dependents"
```

### 7. API Contract Stability
```python
def test_api_contract_stability():
    """Prevent breaking changes to public APIs"""
    current_contracts = extract_api_contracts()
    baseline_contracts = load_baseline_contracts()

    breaking_changes = detect_breaking_changes(baseline_contracts, current_contracts)
    assert len(breaking_changes) == 0, f"Breaking API changes detected: {breaking_changes}"
```

---

## 🔐 Security & Compliance Fitness Functions

### 8. No Hardcoded Secrets
```python
def test_no_hardcoded_secrets():
    """Prevent secrets in source code"""
    for file_path in get_python_files():
        violations = scan_for_secrets(file_path)
        assert len(violations) == 0, f"Hardcoded secrets detected in {file_path}: {violations}"
```

### 9. PII/PHI Isolation
```python
def test_pii_phi_isolation():
    """Ensure PII/PHI handling is properly isolated"""
    pii_handlers = find_pii_handling_code()
    for handler in pii_handlers:
        assert handler.has_encryption, f"PII handler {handler.name} missing encryption"
        assert handler.has_audit_trail, f"PII handler {handler.name} missing audit trail"
```

### 10. Database Access Patterns
```python
def test_database_access_patterns():
    """Prevent raw SQL and ensure repository pattern usage"""
    for file_path in get_business_logic_files():
        sql_queries = find_embedded_sql(file_path)
        assert len(sql_queries) == 0, f"Embedded SQL detected in {file_path}: {sql_queries}"
```

---

## 📊 Performance Fitness Functions

### 11. Import Time Limits
```python
def test_import_time_limits():
    """Ensure modules import quickly"""
    for module in get_modules():
        import_time = measure_import_time(module)
        assert import_time <= 2.0, f"Module {module} imports too slowly: {import_time}s"
```

### 12. Memory Usage Bounds
```python
def test_memory_usage_bounds():
    """Prevent memory leaks in core components"""
    for component in get_core_components():
        memory_usage = measure_memory_usage(component)
        assert memory_usage <= MAX_MEMORY_MB, f"Component {component} uses too much memory: {memory_usage}MB"
```

---

## 🧪 Test Quality Fitness Functions

### 13. Test Coverage Thresholds
```python
def test_coverage_thresholds():
    """Ensure adequate test coverage"""
    for module in get_business_logic_modules():
        coverage = calculate_test_coverage(module)
        assert coverage >= 80, f"Insufficient test coverage for {module}: {coverage}%"
```

### 14. Test Isolation
```python
def test_isolation():
    """Ensure tests don't have hidden dependencies"""
    test_dependencies = analyze_test_dependencies()
    for test_file in get_test_files():
        external_deps = test_dependencies[test_file]
        assert len(external_deps) <= 5, f"Test {test_file} has too many dependencies: {external_deps}"
```

---

## 🔄 CI/CD Integration

### Pre-commit Hooks
```yaml
repos:
  - repo: local
    hooks:
      - id: architectural-fitness
        name: Architectural Fitness Functions
        entry: python -m pytest tests/architecture/ --tb=short
        language: system
        pass_filenames: false
        always_run: true
```

### GitHub Actions Workflow
```yaml
name: Architectural Validation
on: [push, pull_request]

jobs:
  architecture-fitness:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Fitness Functions
        run: |
          python -m pytest tests/architecture/ \
            --junitxml=reports/architecture-fitness.xml \
            --html=reports/architecture-fitness.html

      - name: Upload Reports
        uses: actions/upload-artifact@v3
        with:
          name: architecture-reports
          path: reports/
```

---

## 📈 Monitoring & Alerting

### Dashboard Metrics
1. **Coupling Trend:** Track coupling scores over time
2. **Violation Count:** Monitor connascence violations
3. **Technical Debt:** Measure architectural debt accumulation
4. **Test Coverage:** Track coverage across modules

### Alert Thresholds
- **Critical:** God Object creation (>500 LOC)
- **High:** Cross-module strong connascence introduction
- **Medium:** Coupling score increase >20%
- **Low:** Magic literal density increase >10%

---

## 🛠️ Implementation Commands

### Setup Fitness Functions
```bash
# Create architecture test directory
mkdir -p tests/architecture

# Install dependencies
pip install pytest pytest-html pytest-cov

# Create fitness function test suite
cp architectural_fitness_template.py tests/architecture/test_fitness_functions.py

# Add to CI pipeline
cp .github/workflows/architecture.yml .github/workflows/
```

### Run Fitness Functions
```bash
# Run all architectural tests
pytest tests/architecture/ -v

# Run specific fitness function
pytest tests/architecture/test_fitness_functions.py::test_no_god_objects -v

# Generate HTML report
pytest tests/architecture/ --html=reports/fitness-report.html
```

### Baseline Creation
```bash
# Create baseline measurements
python scripts/create_architecture_baseline.py

# Update baseline after approved changes
python scripts/update_architecture_baseline.py --approve
```

---

## 📋 Fitness Function Maintenance

### Monthly Reviews
1. **Threshold Adjustment:** Review and adjust thresholds based on team capacity
2. **New Function Addition:** Add fitness functions for new architectural concerns
3. **Deprecated Function Removal:** Remove outdated or irrelevant checks
4. **Performance Optimization:** Optimize slow-running fitness functions

### Quarterly Assessments
1. **Effectiveness Analysis:** Measure fitness function effectiveness in preventing issues
2. **Coverage Gaps:** Identify architectural concerns not covered by current functions
3. **Tool Enhancement:** Upgrade tooling and reporting capabilities
4. **Team Training:** Provide training on new fitness functions and architectural principles

---

**Created:** August 21, 2025
**Next Review:** September 21, 2025
**Owner:** Architecture Team
