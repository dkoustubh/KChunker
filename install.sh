#!/bin/bash
# KChunker Installer for macOS and Linux (Ubuntu)
# Sets up the virtual environment and installs all dependencies using uv.

set -e

cd "$(dirname "$0")"

echo "===================================================="
echo "            KCHUNKER INSTALLER (macOS/Linux)        "
echo "===================================================="
echo ""

# Check if uv is installed, if not, install it
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for current terminal session
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv package manager is already installed."
fi

echo "Synchronizing project dependencies..."
uv sync

echo ""
echo "===================================================="
echo "Installation complete! You can now run KChunker."
echo "===================================================="
