# Workflow Fixes

## api-docs.yml
- Indented heredoc content for HTML generation within `run` block.
- Adjusted closing tags and EOF marker to satisfy YAML parser.
- Validated syntax with `python -c "import yaml; yaml.safe_load(open('.github/workflows/api-docs.yml'))"`.

## compression-tests.yml
- Indented Python heredocs and inline scripts so YAML parser treats them as part of the `run` blocks.
- Rewrote benchmark step to use a heredoc-generated `basic_benchmark.py` script.
- Validated with `python -c "import yaml; yaml.safe_load(open('.github/workflows/compression-tests.yml'))"`.

## performance.yml
- Indented multiline commit message within heredoc to fix parsing errors.
- Validated with `python -c "import yaml; yaml.safe_load(open('.github/workflows/performance.yml'))"`.

## test-and-deploy.yml
- Indented Python heredocs and calculation script inside `python -c` blocks.
- Validated with `python -c "import yaml; yaml.safe_load(open('.github/workflows/test-and-deploy.yml'))"`.

## update-stats.yml
- Indented markdown heredoc and commit message block.
- Validated with `python -c "import yaml; yaml.safe_load(open('.github/workflows/update-stats.yml'))"`.

## actionlint
- Attempted to install `actionlint` via `pip`, but package not found.
