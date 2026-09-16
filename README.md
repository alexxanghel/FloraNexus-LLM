# FloraNexus LLM — Grounded Medicinal-Plant Assistant

This repository contains the **LLM / conversational AI component** developed for **FloraNexus**, a team application created for the **Severin Bumbaru 2026 competition**.

FloraNexus combines image-based medicinal-plant recognition with a conversational assistant called **Flora**. The CNN identifies the plant from an image; Flora then explains the detected plant using structured information from a Neo4j knowledge graph.

This repository does **not** contain the complete FloraNexus application. It isolates my primary contribution to the team project: the LLM agent, its tool-calling and grounding layer, prompt/observability integration, and the contract used to consume CNN predictions.

## Project context and contribution

FloraNexus was developed as a team project. My main responsibility was the **LLM/chatbot subsystem**.

The work represented in this repository includes:

- designing the Flora conversational-agent flow;
- consuming structured predictions produced by the CNN pipeline;
- resolving the detected plant against the Neo4j knowledge graph;
- implementing function/tool calling for grounded plant information;
- implementing the Neo4j retrieval layer used by the agent;
- integrating Groq-hosted LLM inference;
- adding an Ollama-compatible provider for local inference;
- managing prompts through Langfuse;
- tracing agent turns, model generations and tool calls with Langfuse;
- maintaining lightweight conversation state and recent chat history;
- validating the JSON contract exchanged between the CNN and the assistant.

The mobile application, CNN training pipeline, administration interface and other application components are outside the scope of this repository.

## What Flora does

Flora does **not** classify plant images itself.

The interaction is split into two stages:

1. the CNN receives an image and produces a structured plant prediction;
2. Flora receives that prediction and answers follow-up questions about the detected plant.

The agent is therefore grounded in two sources of runtime context:

- the current CNN prediction;
- structured facts retrieved from Neo4j through explicit tools.

This separation prevents the LLM from replacing the image classifier and allows the conversational layer to focus on explanation and retrieval.

## Architecture

```text
Plant image
    │
    ▼
CNN classifier
    │
    │ PredictionPayload
    ▼
Flora LLM Agent
    │
    ├── Prediction context
    ├── Session state
    ├── Langfuse-managed prompts
    └── Tool catalog
            │
            ▼
      Neo4j knowledge graph
            │
            ├── identity
            ├── benefits
            ├── uses
            ├── contraindications
            ├── warnings / adverse effects
            ├── interactions
            └── usable plant parts
```

## Agent orchestration

The central component is `agent.py`.

It is responsible for:

- creating the runtime conversation context;
- loading the system prompt from Langfuse;
- injecting CNN prediction data into the model context;
- injecting the resolved plant state into the model context;
- exposing the available tools to the LLM;
- executing tool calls requested by the model;
- returning tool results to the model;
- limiting the number of tool-execution rounds;
- preserving recent user/assistant messages;
- returning the final assistant response.

The agent supports multiple LLM providers through a small provider abstraction.

### Groq

`GroqChatClient` uses the Groq chat-completions API and supports automatic tool calling.

The original configuration uses:

```text
llama-3.1-8b-instant
```

The model can be changed through environment configuration.

### Ollama

`OllamaChatClient` provides an alternative path for running a compatible model locally through Ollama.

This makes the orchestration layer independent from a single hosted LLM provider.

## Grounded tool calling

Flora does not rely only on information stored in the language model.

The agent exposes a set of explicit tools backed by Neo4j:

| Tool | Purpose |
|---|---|
| `get_plant_identity` | Common name, scientific name, family and aliases |
| `get_plant_benefits` | Grounded benefit information |
| `get_plant_uses` | Practical/traditional uses |
| `get_plant_contraindications` | Contraindications, warnings and adverse effects |
| `get_plant_interactions` | Known interaction information |
| `get_plant_usable_parts` | Plant parts used in practice |
| `resolve_plant_candidates` | Alternative candidates from the CNN top-k output |

Each tool has a structured JSON schema that is exposed to the LLM.

When the model requests a tool, Flora:

1. validates the requested tool;
2. resolves the current plant if necessary;
3. executes the relevant Neo4j query;
4. compacts the returned data;
5. sends the grounded result back to the model;
6. allows the model to formulate a conversational response.

## Neo4j grounding layer

`flora_store.py` contains the data-access layer for the knowledge graph.

The LLM itself does not generate Cypher queries. Instead, the application exposes predefined retrieval operations.

This layer handles:

- plant resolution from the CNN identifier, Romanian name, scientific name and aliases;
- plant identity retrieval;
- benefits;
- uses;
- contraindications;
- warnings;
- adverse effects;
- interactions;
- usable parts;
- alternative plant candidates.

The store also filters records based on fields such as `chatbot_visible` and `record_status` before they reach the LLM.

## Connecting the CNN and the LLM

`prediction_payload.py` defines the typed contract between the image classifier and Flora.

A valid prediction contains:

- request identifier;
- timestamp;
- image/source metadata;
- predicted plant identifier;
- Romanian plant name;
- scientific plant name;
- confidence score;
- top-k candidates;
- model metadata;
- inference time;
- prediction status.

Example:

```json
{
  "request_id": "demo-001",
  "timestamp": "2026-01-01T12:00:00Z",
  "input": {
    "image_path": "example.jpg",
    "source": "demo"
  },
  "prediction": {
    "plant_id": "matricaria_chamomilla",
    "plant_name_ro": "Mușețel",
    "plant_name_scientific": "Matricaria chamomilla",
    "confidence": 0.93,
    "top_k": [
      {"plant_id": "matricaria_chamomilla", "score": 0.93},
      {"plant_id": "leucanthemum_vulgare", "score": 0.04},
      {"plant_id": "tanacetum_vulgare", "score": 0.03}
    ]
  },
  "inference": {
    "model_name": "efficientnet_b3",
    "model_version": "1.0",
    "inference_ms": 42
  },
  "status": "predicted"
}
```

The agent resolves this external prediction to the internal `graph_plant_id` used by Neo4j before answering plant-specific questions.

## Prompt management

The original Flora implementation does not hardcode its main prompts in the Python source.

They are loaded from **Langfuse** using three prompt names:

```text
flora_behavior
flora_tools
flora_style_ro
```

This allows behavior, tool-use instructions and Romanian response style to be versioned separately from the application code.

The actual prompt contents are not present in the source archive used to prepare this repository, so this repository contains the **prompt-loading integration**, not exported Langfuse prompt definitions.

## Observability

Langfuse is also used for tracing.

The implementation records separate observations for:

- an entire Flora agent turn;
- individual LLM generations;
- individual tool calls.

This makes it possible to inspect which context was sent, which tools were selected and how the final answer was produced during development.

## Conversation state

Flora keeps lightweight runtime state instead of relying on an unlimited chat history.

The current session stores the resolved plant, while the agent keeps only a limited number of recent user/assistant turns in the model context.

This reduces unnecessary prompt growth while keeping enough context for short follow-up questions.

## Repository structure

```text
FloraNexus-LLM/
├── examples/
│   ├── run_chat.py
│   └── sample_prediction.json
│
├── src/floranexus_llm/
│   ├── __init__.py
│   ├── agent.py
│   ├── flora_store.py
│   ├── flora_tools.py
│   ├── prediction_payload.py
│   └── settings.py
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Installation

Python 3.11 is recommended.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

Copy-Item .env.example .env
```

Then configure the Neo4j, Langfuse and LLM-provider credentials in `.env`.

## Configuration

The main configuration options are:

```text
NEO4J_URI
NEO4J_DATABASE
NEO4J_USERNAME
NEO4J_PASSWORD

FLORANEXUS_LLM_PROVIDER
FLORANEXUS_LLM_BASE_URL
FLORANEXUS_LLM_API_KEY
FLORANEXUS_LLM_MODEL
FLORANEXUS_LLM_TEMPERATURE
FLORANEXUS_LLM_TIMEOUT_S
FLORANEXUS_LLM_MAX_TOOL_ROUNDS

GROQ_API_KEY

LANGFUSE_SECRET_KEY
LANGFUSE_PUBLIC_KEY
LANGFUSE_BASE_URL

FLORANEXUS_PROMPT_BEHAVIOR
FLORANEXUS_PROMPT_TOOLS
FLORANEXUS_PROMPT_STYLE
```

Real credentials must never be committed to GitHub. The repository includes only `.env.example`.

## Running the local chat

After configuring Neo4j, Langfuse and the selected LLM provider:

```bash
python examples/run_chat.py
```

The example passes a mock CNN prediction to Flora and starts an interactive terminal conversation.

Type `exit`, `quit` or `q` to stop the session.

## Technologies

- Python
- Large Language Models
- Groq
- Ollama
- Neo4j
- Cypher
- Langfuse
- Pydantic
- Tool / function calling
- Structured JSON contracts

## Important notes

- This repository represents the LLM/chatbot contribution to the larger FloraNexus team project.
- Flora is a conversational layer; plant-image identification is performed by the separate CNN component.
- The Neo4j database contents are not included in this repository.
- Langfuse prompt text is not included because it was stored externally.
- `.env` credentials are intentionally excluded.
- Information returned by the assistant depends on the contents and quality of the connected knowledge graph.
- The assistant should not be treated as a substitute for professional medical or botanical advice.

## Project summary

**Project:** FloraNexus  
**Event:** Severin Bumbaru 2026  
**Repository scope:** LLM / conversational-agent contribution  
**Assistant:** Flora  
**Grounding:** Neo4j knowledge graph  
**Prompt management & tracing:** Langfuse  
**LLM providers implemented:** Groq and Ollama  
**Integration input:** structured CNN prediction payload
