#!/bin/bash
# KChunker Dashboard Launcher for macOS
# Double-clicking this file opens Terminal, prompts for a file path, and launches the Dear PyGui dashboard.

# Navigate to the script's directory so imports and virtual environment resolve correctly
cd "$(dirname "$0")"

clear
echo "===================================================="
echo "            KCHUNKER GUI AUTO-LAUNCHER              "
echo "===================================================="
echo ""
echo "Drag-and-drop a file here (or type the file address),"
echo "then press Enter to launch the dashboard and auto-ingest:"
echo ""
read -r filepath

# Clean up macOS terminal drag-and-drop artifacts (stripping surrounding quotes and escaping backslashes)
filepath=$(echo "$filepath" | sed -e "s/^'//" -e "s/'$//" -e 's/^"//' -e 's/"$//')
filepath="${filepath//\\ / }"

if [ -n "$filepath" ]; then
    echo ""
    echo "Starting GUI dashboard with auto-ingestion for:"
    echo "  $filepath"
    echo ""
    PYTHONPATH=. .venv/bin/python gui.py --file "$filepath"
else
    echo ""
    echo "Launching GUI dashboard..."
    echo ""
    PYTHONPATH=. .venv/bin/python gui.py
fi
