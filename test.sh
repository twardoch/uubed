#!/usr/bin/env bash
# this_file: test.sh
set -euo pipefail
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec uvx hatch test
