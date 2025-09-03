#!/bin/bash

echo "🔍 Running local placeholder validation (matching CI exactly)..."

# Define patterns to check (matching CI workflow)
declare -a PLACEHOLDER_PATTERNS
PLACEHOLDER_PATTERNS[0]="T""ODO:"
PLACEHOLDER_PATTERNS[1]="F""IXME:"
PLACEHOLDER_PATTERNS[2]="X""XX:"
PLACEHOLDER_PATTERNS[3]="H""ACK:"
PLACEHOLDER_PATTERNS[4]="place""holder"
PLACEHOLDER_PATTERNS[5]="not ""implemented"
PLACEHOLDER_PATTERNS[6]="st""ub"
PLACEHOLDER_PATTERNS[7]="mo""ck"
PLACEHOLDER_PATTERNS[8]="fa""ke"
PLACEHOLDER_PATTERNS[9]="dum""my"
PLACEHOLDER_PATTERNS[10]="tempor""ary"
PLACEHOLDER_PATTERNS[11]="temp ""implementation"
PLACEHOLDER_PATTERNS[12]="coming ""soon"
PLACEHOLDER_PATTERNS[13]="to be ""implemented"

# Files to check (matching CI exclusions)
FILES_TO_CHECK=$(find . -type f \( \
  -name "*.go" -o \
  -name "*.rs" -o \
  -name "*.py" -o \
  -name "*.proto" -o \
  -name "*.yaml" -o \
  -name "*.yml" \
  \) \
  ! -path "./tests/*" \
  ! -path "./*test*" \
  ! -path "./docs/*" \
  ! -path "./examples/*" \
  ! -path "./.git/*" \
  ! -path "./target/*" \
  ! -path "./vendor/*" \
  ! -path "./.claude/*" \
  ! -path "./tools/development/*" \
  ! -path "./tools/*/development/*" \
  ! -path "./archive/*" \
  ! -path "./*/archive/*" \
  ! -path "./*/deprecated/*" \
  ! -path "./*/legacy/*" \
  ! -path "./*/site-packages/*" \
  ! -path "./**/site-packages/*" \
  ! -path "./node_modules/*" \
  ! -path "./**/node_modules/*" \
  ! -path "./infrastructure/shared/tools/stub_*" \
  ! -path "./infrastructure/twin/quality/stub_*" \
  ! -path "./experiments/*" \
  ! -path "./swarm/*" \
  ! -path "./benchmarks/*" \
  ! -path "./*/benchmarks/*" \
  ! -path "./**/__pycache__/*" \
  ! -path "./**/.mypy_cache/*" \
  ! -path "./**/venv/*" \
  ! -path "./**/env/*" \
  ! -path "./build/*" \
  ! -path "./dist/*" \
  ! -path "./*/build/*" \
  ! -path "./*/dist/*" \
  ! -name "*.generated.*" \
  ! -name "*_pb.go" \
  ! -name "*_grpc.pb.go")

VIOLATIONS_FOUND=false

for pattern in "${PLACEHOLDER_PATTERNS[@]}"; do
  echo "Checking for pattern: '$pattern'"

  # Check each file individually
  while IFS= read -r file; do
    if grep -l -i "$pattern" "$file" 2>/dev/null; then
      # Skip legitimate development utilities and non-production files
      if [[ "$file" =~ (stub_elimination_system|stub_fix|list_stubs|test_stub) ]] || \
         [[ "$file" =~ \.claude/ ]] || \
         [[ "$file" =~ tools/development/ ]] || \
         [[ "$file" =~ archive/ ]] || \
         [[ "$file" =~ deprecated/ ]] || \
         [[ "$file" =~ legacy/ ]] || \
         [[ "$file" =~ site-packages/ ]] || \
         [[ "$file" =~ node_modules/ ]] || \
         [[ "$file" =~ __pycache__/ ]] || \
         [[ "$file" =~ /venv/ ]] || \
         [[ "$file" =~ /env/ ]] || \
         [[ "$file" =~ build/ ]] || \
         [[ "$file" =~ dist/ ]] || \
         [[ "$file" =~ benchmarks/ ]] || \
         [[ "$file" =~ scripts/ ]] || \
         [[ "$file" =~ tools/ ]] || \
         [[ "$file" =~ \.example$ ]] || \
         [[ "$file" =~ \.template$ ]] || \
         [[ "$file" =~ \.bak$ ]] || \
         [[ "$file" =~ _pb\.go$ ]] || \
         [[ "$file" =~ _grpc\.pb\.go$ ]] || \
         [[ "$file" =~ \.generated\. ]]; then
        echo "[INFO] Skipping non-production file: $file"
        continue
      fi
      
      # Check for legitimate technical terms that should be excluded
      if echo "$file" | grep -q "$pattern"; then
        # Check for legitimate gRPC stub usage or similar technical terms
        if [[ "$pattern" =~ "stub" ]] && grep -q -E "(grpc.*stub|stub.*service|service.*stub|rpc.*stub)" "$file"; then
          echo "[INFO] Skipping legitimate gRPC stub usage in: $file"
          continue
        fi
        # Check for legitimate mock usage in interface definitions
        if [[ "$pattern" =~ "mock" ]] && grep -q -E "(interface.*mock|mock.*interface|type.*mock)" "$file"; then
          echo "[INFO] Skipping legitimate mock interface in: $file"
          continue
        fi
      fi
      echo "[FAIL] Found placeholder pattern '$pattern' in production code: $file"
      grep -n -i "$pattern" "$file" 2>/dev/null || true
      VIOLATIONS_FOUND=true
    fi
  done <<< "$FILES_TO_CHECK"
done

if [ "$VIOLATIONS_FOUND" = true ]; then
  echo ""
  echo "[FAIL] PLACEHOLDER VALIDATION FAILED"
  echo "Production code contains placeholder patterns that must be removed."
  echo "Please implement all functionality before merging."
  exit 1
else
  echo "[PASS] PLACEHOLDER VALIDATION PASSED"
  echo "No placeholder patterns found in production code."
fi