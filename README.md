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
Ensure you have `uv` installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the dependencies:
```bash
uv sync
```

## Running the CLI
```bash
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

2. **Project Root Shortcut Script**:
   ```bash
   ./gui
   # Or with auto-ingestion:
   ./gui --file <path_to_document>
   ```

3. **Package Manager Script Entrypoint**:
   ```bash
   uv run kchunker-gui
   ```

4. **macOS Double-Clickable Command**:
   Simply double-click the `launch_gui.command` file in Finder. It will open Terminal, let you drag-and-drop a file path, and start the GUI dashboard.

