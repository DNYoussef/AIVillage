#!/bin/bash
set -euo pipefail
mkdir -p tmp_submission/mixnode
cargo bench -p betanet-mixnode >/tmp/bench.log
cp target/criterion/sphinx_parse/new/estimates.json tmp_submission/mixnode/bench.json
