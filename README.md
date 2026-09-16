# FloraNexus LLM — Medicinal Plant Assistant

This repository contains my **LLM contribution** to **FloraNexus**, a team project developed for the **Severin Bumbaru 2026 competition**.

FloraNexus was designed as an EcoLocation-style application for identifying medicinal plants found in Galați County and providing useful information about them.

The complete application combined:

- CNN-based plant identification;
- medicinal plant information;
- user-created points of interest;
- location-based functionality;
- an interactive map;
- an AI-powered conversational assistant.

This repository does **not** contain the entire FloraNexus application.

It contains only my contribution to the project: the **LLM-based conversational assistant called Flora**, together with the logic required to connect the language model to the plant knowledge base and to the predictions produced by the CNN classifier.

---

## Project authorship and scope

FloraNexus was developed as a team project.

My main responsibility was the **LLM and conversational AI component**.

I worked on:

- implementing the Flora conversational agent;
- integrating **GPT-OSS-120B through OpenRouter**;
- implementing LLM tool calling;
- connecting the agent to the Neo4j plant knowledge graph;
- retrieving grounded plant information;
- handling benefits and medicinal uses;
- retrieving usable plant parts;
- retrieving contraindications;
- retrieving warnings and adverse effects;
- retrieving possible interactions;
- integrating the CNN prediction results into the chatbot context;
- handling alternative CNN predictions when the detected plant is questioned;
- implementing lightweight conversation-session memory;
- integrating Langfuse for prompt management and tracing;
- defining structured payloads between the CNN classifier and the LLM component.

Other components of FloraNexus, such as the CNN training pipeline, mobile application, backend infrastructure, interactive map, authentication and administration interface, were developed separately and are not included in this repository.

---

## What is Flora?

**Flora** is the conversational assistant integrated into FloraNexus.

After the CNN component identifies a medicinal plant from an image, Flora receives the prediction and allows the user to ask questions about that plant.

For example, the assistant can provide information about:

- plant identity;
- medicinal benefits;
- traditional or common uses;
- usable plant parts;
- contraindications;
- warnings;
- adverse effects;
- possible interactions.

Instead of depending only on the general knowledge of the language model, Flora retrieves plant-specific information from a **Neo4j knowledge graph**.

This makes the responses more grounded in the information stored by the application.

---

## LLM architecture

The conversational pipeline uses **GPT-OSS-120B through OpenRouter**.

The language model is responsible for understanding the user's question and deciding which plant-information tools should be called.

The relevant information is then retrieved from Neo4j and used to generate the final response.

The general flow is:

```text
User question
      ↓
Flora Agent
      ↓
GPT-OSS-120B
through OpenRouter
      ↓
Tool selection
      ↓
Neo4j Knowledge Graph
      ↓
Plant information retrieval
      ↓
Grounded LLM response
```

This tool-based approach allows the model to retrieve structured information rather than relying exclusively on its pretrained knowledge.

---

## CNN integration

The LLM component is designed to work together with the FloraNexus CNN classifier.

The CNN predicts the plant species from an uploaded image and provides structured prediction data to the LLM component.

A prediction payload can contain information such as:

```json
{
  "prediction": {
    "label": "musetel",
    "plant_name_ro": "Mușețel",
    "plant_name_scientific": "Matricaria chamomilla",
    "confidence": 0.93
  },
  "top_k": [
    {
      "label": "musetel",
      "confidence": 0.93
    },
    {
      "label": "margareta",
      "confidence": 0.04
    }
  ]
}
```

Flora uses this context when answering questions about the identified plant.

For example:

```text
CNN:
Plant detected → Mușețel

User:
What are its benefits?

Flora:
→ identifies the current plant
→ retrieves the relevant benefit information from Neo4j
→ generates the final response
```

---

## Alternative prediction handling

Image classification is not always perfect.

Because of this, Flora can also work with alternative predictions from the CNN's `top_k` results.

If the user indicates that the detected plant may be incorrect, the assistant can use the alternative prediction candidates when resolving the plant context.

This creates a connection between the uncertainty of the computer-vision model and the conversational layer.

---

## Tool calling

Flora uses LLM tool calling to retrieve specific categories of plant information.

The available tools cover information such as:

### Plant identity

Retrieves basic information about the plant.

### Benefits

Retrieves known medicinal or health-related benefits stored in the knowledge graph.

### Uses

Retrieves the plant's documented or traditional uses.

### Usable parts

Retrieves which parts of the plant can be used, such as:

- leaves;
- flowers;
- roots;
- stems;
- fruits.

### Contraindications

Retrieves situations where the plant should not be used.

### Warnings and adverse effects

Retrieves known warnings or possible adverse effects.

### Interactions

Retrieves known or documented interactions associated with the plant.

The language model decides which tool is relevant based on the user's question.

---

## Neo4j knowledge graph

Plant-specific information is stored in **Neo4j**.

The LLM does not need to keep the complete plant database inside its prompt.

Instead, the agent retrieves only the information relevant to the current request.

Conceptually:

```text
Plant
├── benefits
├── uses
├── usable parts
├── contraindications
├── warnings
└── interactions
```

The agent resolves the plant detected by the CNN to the corresponding entity in the graph before requesting additional information.

---

## Langfuse integration

Langfuse is used for LLM observability and prompt management.

The integration allows the project to track elements such as:

- agent turns;
- LLM generations;
- tool calls;
- model requests;
- conversation traces;
- prompt versions.

The project also loads the prompts used by Flora through Langfuse.

The actual production prompt contents are not included in this repository.

---

## Conversation memory

Flora includes lightweight session memory to preserve the relevant conversation context between user messages.

For example:

```text
User:
What plant is this?

Flora:
This appears to be chamomile.

User:
What are its contraindications?
```

The second message can still be interpreted in the context of the plant currently being discussed.

The conversation history sent to the model is limited so that the prompt does not grow indefinitely.

---

## Repository structure

```text
FloraNexus-LLM/
├── examples/
│   ├── run_chat.py
│   └── sample_prediction.json
│
├── src/floranexus_llm/
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

---

## Main components

### `agent.py`

Contains the main Flora conversational-agent logic.

Responsibilities include:

- communicating with the language model;
- managing conversation context;
- executing tool calls;
- coordinating knowledge retrieval;
- generating the final assistant response.

### `flora_tools.py`

Defines the tools available to the language model.

These tools allow Flora to retrieve structured plant information from the knowledge graph.

### `flora_store.py`

Handles access to the plant knowledge stored in Neo4j.

### `prediction_payload.py`

Defines the structured input received from the CNN classifier.

This creates a clear interface between the image-classification component and the conversational component.

### `settings.py`

Contains the configuration used by the LLM component.

---

## Technologies

- Python
- OpenRouter
- GPT-OSS-120B
- Neo4j
- Langfuse
- Pydantic
- python-dotenv

---

## Installation

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file using `.env.example` as a reference.

Example:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openai/gpt-oss-120b

NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password

LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=your_langfuse_host
```

Do not commit real API keys or credentials to GitHub.

---

## Example interaction

```text
Detected plant:
Mușețel — Matricaria chamomilla

User:
What are the benefits of this plant?

Flora:
→ receives the CNN plant context
→ determines that benefit information is required
→ calls the appropriate plant-information tool
→ retrieves information from Neo4j
→ generates a grounded response
```

Another example:

```text
User:
Are there any contraindications?

Flora:
→ keeps the current plant in conversation context
→ calls the contraindications tool
→ retrieves the relevant Neo4j information
→ returns the response
```

---

## Why use tool calling?

A general-purpose LLM can answer many questions from its internal knowledge, but this can introduce information that is inconsistent with the application's own database.

Flora instead uses a tool-based approach.

The language model is primarily responsible for:

- understanding user intent;
- selecting the required information;
- deciding which tool to call;
- turning retrieved structured data into a natural response.

The plant-specific information is retrieved from the application's knowledge graph.

This separation helps make the conversational assistant more consistent with the data stored by FloraNexus.

---

## Important notes

- This repository contains my LLM contribution to the larger FloraNexus team project.
- It does not contain the complete FloraNexus application.
- The CNN component is maintained separately.
- The plant knowledge graph is required for full functionality.
- The production Langfuse prompts are not included.
- API credentials are not included.
- GPT-OSS-120B is accessed through OpenRouter.
- The chatbot is intended as an informational component of the project.
- Plant-related responses should not be treated as professional medical advice.

---

## Project context

**Project:** FloraNexus  
**Event:** Severin Bumbaru 2026  
**Repository scope:** LLM and conversational AI contribution  
**Assistant:** Flora  
**LLM provider:** OpenRouter  
**Language model:** GPT-OSS-120B  
**Knowledge source:** Neo4j  
**Observability and prompt management:** Langfuse  
**Primary functionality:** conversational information about identified medicinal plants, including benefits, uses, usable parts, contraindications, warnings and interactions
