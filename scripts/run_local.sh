#!/usr/bin/env bash
set -euo pipefail

image-archive init
image-archive index "${1:-$HOME/Pictures}"
image-archive serve
