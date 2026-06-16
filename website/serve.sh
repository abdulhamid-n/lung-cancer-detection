#!/usr/bin/env bash
# Serve the PULMONET site locally. A local HTTP server is required for the
# in-browser ONNX demo (browsers block fetching model.onnx from file://).
#
#   ./serve.sh          # serves on http://localhost:8000
#   ./serve.sh 9000     # custom port
set -e
cd "$(dirname "$0")"
PORT="${1:-8000}"
echo "PULMONET  ->  http://localhost:${PORT}"
echo "(Ctrl+C to stop)"
python3 -m http.server "${PORT}"
