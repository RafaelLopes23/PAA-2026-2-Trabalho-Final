#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="$ROOT_DIR/.local/ollama"

mkdir -p "$INSTALL_DIR" "$ROOT_DIR/.ollama/models"

curl -L https://ollama.com/download/ollama-linux-amd64.tar.zst -o /tmp/ollama-linux-amd64.tar.zst
curl -L https://ollama.com/download/ollama-linux-amd64-rocm.tar.zst -o /tmp/ollama-linux-amd64-rocm.tar.zst

tar --zstd -xf /tmp/ollama-linux-amd64.tar.zst -C "$INSTALL_DIR"
tar --zstd -xf /tmp/ollama-linux-amd64-rocm.tar.zst -C "$INSTALL_DIR"

echo "Ollama instalado em $INSTALL_DIR"
echo "Use: bash scripts/start_local_ollama.sh"
