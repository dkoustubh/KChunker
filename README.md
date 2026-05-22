# KChunker

KChunker is a lightweight, ultra-fast, terminal-first intelligent document chunking engine designed for industrial RFQ processing and Retrieval-Augmented Generation (RAG) systems.

## Core Features
* Automatic document classification and parser routing.
* Semantic, hierarchical, table-aware, layout-aware, and email-thread chunking strategies.
* Metadata preservation and enrichment.
* Embedding generation and local indexing via FAISS and ChromaDB.

## Process Flow Architecture

The diagram below outlines the lifecycle of a document processed through KChunker:

```mermaid
graph TD
    A[Input File / Directory] --> B{Parser Router}
    
    %% Ingestion / Parsing Phase
    B -->|Native PDF| C[PDF Parser]
    B -->|Scanned PDF / Image| D[OCR Parser + PaddleOCR]
    B -->|Excel / CSV| E[Excel Parser]
    B -->|Word Doc| F[DOCX Parser]
    B -->|Email / MSG| G[Email Parser]
    B -->|Plain Text| H[TXT Parser]

    %% Layout & Chunking Phase
    C & D & E & F & G & H --> I[Extracted Document Structure]
    I --> J[Hybrid Chunker]
    
    J -->|Table Blocks| K1[Table-Aware Chunker <br/><i>Markdown + Header Persistence</i>]
    J -->|Sentences/Text| K2[Semantic Chunker <br/><i>Cosine Similarity Split</i>]
    J -->|Layout Hierarchy| K3[Hierarchical Chunker <br/><i>Parent-Child Linking</i>]
    J -->|Email Trails| K4[Email Chunker <br/><i>Trail Split & Linking</i>]

    %% Vectorization & Storage
    K1 & K2 & K3 & K4 --> L[Structured JSON Chunk File Storage]
    K1 & K2 & K3 & K4 --> M[SentenceTransformers Embedding Generator]
    
    M --> N{Vector DB Store}
    N -->|Persistent Store| O1[(ChromaDB)]
    N -->|Flat CPU Index| O2[(FAISS)]

    %% Querying
    O1 & O2 -.-> P[CLI / GUI Search Query]
    P --> Q[Contextually Rich Search Results]
```

## Installation & Setup

### macOS & Ubuntu (Linux)
You can set up KChunker automatically using the installer script, or run manual setup commands:

* **Automatic Shortcut**:
  ```bash
  ./install.sh
  ```
* **Manual Setup**:
  1. Install `uv` if you haven't:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
  2. Sync dependencies:
     ```bash
     uv sync
     ```

### Windows OS
You can install dependencies automatically via the batch installer or perform manual commands:

* **Automatic Shortcut**:
  Double-click `install.bat`
* **Manual Setup**:
  1. Open PowerShell and run to install `uv`:
     ```powershell
     irm https://astral.sh/uv/install.ps1 | iex
     ```
  2. Sync dependencies:
     ```powershell
     uv sync
     ```

---

## Running the CLI

### macOS & Ubuntu (Linux)
```bash
uv run python main.py --file <path_to_document>
```

### Windows OS
```cmd
uv run python main.py --file <path_to_document>
```

## Running the GUI Dashboard
You can launch the interactive Dear PyGui dashboard using any of the following shortcuts:

1. **CLI Flag Shortcut**:
   ```bash
   uv run python main.py --gui
   # Or with auto-ingestion:
   uv run python main.py --gui --file <path_to_document>
   ```

2. **Project Root Shortcut Script (macOS / Linux)**:
   ```bash
   ./gui
   # Or with auto-ingestion:
   ./gui --file <path_to_document>
   ```

3. **Package Manager Script Entrypoint**:
   ```bash
   uv run kchunker-gui
   ```

4. **Double-Clickable GUI Shortcuts**:
   * **macOS**: Double-click `launch_gui.command`. It opens Terminal, prompts for a file path (optional), and runs the dashboard.
   * **Windows OS**: Double-click `launch_gui.bat`. It opens the Command Prompt, prompts for a file path (optional), and launches the dashboard.
   * **Ubuntu / Linux**: Run `./launch_gui.sh` or double-click it. It opens Terminal, prompts for a file path (optional), and starts the dashboard.


