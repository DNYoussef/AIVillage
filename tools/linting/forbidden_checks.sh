#!/usr/bin/env bash
# Copyright (c) 2024 AI Village
# Licensed under the MIT License.

set -euo pipefail
PATTERN='TODO|FIXME|stub|not implemented|unimplemented!|panic!|expect|unwrap'
# search repository excluding tmp_submission directory
if rg --hidden --glob '!tmp_submission/*' --ignore-case --regexp "$PATTERN" -n .; then
  echo "Forbidden terms found" >&2
  exit 1
fi
