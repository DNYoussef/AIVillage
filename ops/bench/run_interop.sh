#!/bin/bash
set -euo pipefail
mkdir -p tmp_submission/mixnode
cargo test -p betanet-mixnode --test interop_route -- --nocapture > tmp_submission/mixnode/interop.log
