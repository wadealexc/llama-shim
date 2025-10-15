#!/usr/bin/env bash
set -euo pipefail

# Set up environment for use with systemd
# Note that your nvm/llama-server locations might differ.

## load nvm so we can call `exec node ...`
source "$HOME/.nvm/nvm.sh"
## place `llama-server` in PATH
export PATH="$HOME/llama.cpp/build/bin:$PATH"

exec node dist/index.js