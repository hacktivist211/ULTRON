# IRIS: Local Self-Corrective Voice Assistant & Memory Architecture

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C)
![llama.cpp](https://img.shields.io/badge/llama.cpp-Qwen3.5--4B-green)
![CosyVoice2](https://img.shields.io/badge/TTS-CosyVoice2-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

IRIS is a low-latency, resource-disciplined, local-first AI assistant designed for hardware-constrained environments (specifically, a single 6 GB VRAM GPU). It combines **llama.cpp** for inference, **CosyVoice2** for zero-shot voice cloning, and a **dual-tier SQLite memory system** to provide conversational continuity and self-corrective RAG (CRAG).

This document details the architecture, the strict single-GPU resource topology, the incremental streaming pipeline, and the memory topology.

---

## 1. System Architecture

IRIS is built on the **"Single Inference Runtime"** principle. To prevent out-of-memory (OOM) crashes on a 6 GB RTX 4050, only one LLM instance is ever loaded into VRAM. The voice agent, the CRAG research pipeline, and the memory embedding engine all act as lightweight HTTP clients to the same `llama-server`.

### 1.1 High-Level Topology

```mermaid
graph TD
    subgraph "GPU (6GB VRAM Limit)"
        LLM[llama.cpp Server<br/>Qwen3.5-4B Q8_0<br/>--jinja --embeddings --pooling mean]
        TTS[CosyVoice2 TTS<br/>FP16 Lazy-Loaded]
    end

    subgraph "External"
        Tavily[Tavily Web Search API]
    end

    subgraph "Primary Runtime (Voice)"
        Jarvis[jarvis.py / iris_boot.py]
        ToolRuntime[tool_runtime.py<br/>Incremental SSE Client]
        VoiceCache[voice_cache.py<br/>Frontend Memoization]
    end

    subgraph "CRAG Stack (Research)"
        Main[main.py]
        Router[evaluator_router.py<br/>Adaptive Routing & Grading]
        Generator[generator.py<br/>Prompt Assembly]
    end

    subgraph "Storage (Flat Files)"
        MemStore[(memory.db<br/>Personal Facts & Dense Vectors)]
        TempStore[(temporal_memory.db<br/>Episodic Q&A Logs)]
        VectorStore[(crag_vector_store<br/>ChromaDB Documents)]
    end

    %% Connections
    Jarvis <--> ToolRuntime
    Jarvis <--> VoiceCache
    VoiceCache --> TTS
    
    Main <--> Router
    Main <--> Generator
    
    ToolRuntime <-->|SSE /v1/chat/completions| LLM
    ToolRuntime <-->|/v1/embeddings| LLM
    ToolRuntime <-->|HTTP| Tavily
    
    Router <-->|HTTP Temp 0| LLM
    Generator <-->|HTTP SSE| LLM
    
    ToolRuntime <--> MemStore
    Main <--> TempStore
    Main <--> VectorStore
