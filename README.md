# Agentic AI Developer Roadmap (2026)

[![Roadmap](https://img.shields.io/badge/Roadmap-2026-green.svg)](https://github.com/your-org/agentic-ai-roadmap)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A structured, engineering-focused roadmap for developers building production AI agent systems.

Inspired by the [DotNet Developer Roadmap](https://github.com/milanm/DotNet-Developer-Roadmap).

## Disclaimer

> This roadmap represents a practical learning path based on real production patterns. The AI agent ecosystem moves fast, tool recommendations reflect what works today, not endorsements. Use your judgment.

## The Roadmap

Download the [PDF version](Agentic%20AI%20Developer%20Roadmap.pdf) with **clickable resource links**.

[![Agentic AI Developer Roadmap](Agentic%20AI%20Developer%20Roadmap.png)](Agentic%20AI%20Developer%20Roadmap.pdf)

---

## Table of Contents

- [Level 0 — Foundations](#level-0--foundations)
- [Level 1 — LLM Foundations](#level-1--llm-foundations)
- [Level 2 — Single-Agent Systems](#level-2--single-agent-systems)
- [Level 3 — Tool Design & External Integrations](#level-3--tool-design--external-integrations)
- [Level 4 — Retrieval, Planning & Advanced Reasoning](#level-4--retrieval-planning--advanced-reasoning)
- [Level 5 — Memory & State](#level-5--memory--state)
- [Level 6 — Multi-Agent Orchestration](#level-6--multi-agent-orchestration)
- [Level 7 — Production & Operations](#level-7--production--operations)
- [Level 8 — Capstone Project](#level-8--capstone-project)
- [Learning Path](#recommended-learning-path)
- [Architectural Patterns](#key-architectural-patterns)
- [Glossary](#glossary)
- [Related Roadmaps](#related-roadmaps)

---

## Level 0 — Foundations

Complete this level first. These skills are assumed throughout the rest of the roadmap.

| Skill | Why You Need It | Quick Resources |
|---|---|---|
| **TypeScript / Node.js (intermediate+)** | Primary language for AI agent development | [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/) · [Node.js Learn](https://nodejs.org/en/learn) · [Total TypeScript (Matt Pocock)](https://www.totaltypescript.com/) |
| **REST APIs & HTTP** | Agents call LLM APIs and tool APIs over HTTP | [HTTP & REST overview (MDN)](https://developer.mozilla.org/en-US/docs/Web/HTTP) · [Express.js Guide](https://expressjs.com/en/guide/routing.html) |
| **Async Programming** | Agent loops, parallel tool calls, streaming all require async | [MDN: Async/Await](https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Asynchronous) · [Node.js Streams](https://nodejs.org/api/stream.html) |
| **Git & CLI** | Version control, prompt versioning, CI/CD pipelines | [Pro Git Book](https://git-scm.com/book/en/v2) |
| **Docker basics** | Sandboxing tools, deploying agents, local dev environments | [Docker Getting Started](https://docs.docker.com/get-started/) · [Docker Compose](https://docs.docker.com/compose/) |
| **SQL fundamentals** | Many agents interact with databases as tools | [SQLBolt](https://sqlbolt.com/) |
| **JSON & JSON Schema** | Tool definitions, structured outputs, API contracts | [JSON Schema Guide](https://json-schema.org/learn/getting-started-step-by-step) |
| **AI Development Tools** | Modern AI-powered development environments and assistants | [GitHub Copilot](https://github.com/features/copilot) · [Cursor](https://www.cursor.com/) · [Windsurf](https://codeium.com/windsurf) · [Replit Agent](https://replit.com/ai) |

---

## Level 1 — LLM Foundations

Before building agents, understand how LLMs work at a systems level.

### Core Concepts

- [ ] How transformer-based LLMs generate text (autoregressive decoding)
- [ ] Tokens, context windows (now 128k-2M tokens), and their cost/latency implications
- [ ] Reasoning models (o1, o3, DeepSeek R1): chain-of-thought, max reasoning tokens/effort controls
- [ ] Model selection: reasoning vs fast vs extended-context models — cost/latency/quality tradeoffs
- [ ] Temperature, top-p, frequency penalty — what they actually control
- [ ] System prompts vs. user prompts vs. assistant messages
- [ ] Chat completion API contract (messages array, roles, tool definitions)
- [ ] Structured output: JSON mode, function calling schemas, constrained decoding
- [ ] Streaming responses and server-sent events (SSE)
- [ ] Streaming tool calls and parallel tool execution
- [ ] Prompt caching: system prompt caching, context prefix caching, cost reduction strategies
- [ ] Rate limits, retries, and backoff strategies for LLM APIs
- [ ] Extended context strategies: when to use 200k+ context vs RAG vs summarization

### Resources

| Topic | Links |
|---|---|
| **LLM Fundamentals** | [What Is a Large Language Model?](https://developers.google.com/machine-learning/resources/intro-llms) · [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) · [Andrej Karpathy: Intro to LLMs (Video)](https://www.youtube.com/watch?v=zjkBMFhNj_g) · [3Blue1Brown: Neural Networks (Video)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) |
| **Reasoning Models** | [OpenAI o1 System Card](https://openai.com/index/openai-o1-system-card/) · [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/) · [DeepSeek R1](https://github.com/deepseek-ai/DeepSeek-R1) · [Reasoning Tokens & Effort Controls](https://platform.openai.com/docs/guides/reasoning) |
| **Prompt Engineering** | [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) · [Anthropic Prompt Engineering Docs](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview) · [Brex's Prompt Engineering Guide](https://github.com/brexhq/prompt-engineering) · [ChatGPT Prompt Engineering for Developers (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) · [Prompt Engineering Guide](https://www.promptingguide.ai/) |
| **Structured Output** | [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) · [Instructor Library](https://github.com/instructor-ai/instructor) · [Outlines (Constrained Generation)](https://github.com/dottxt-ai/outlines) |
| **APIs & SDKs** | [OpenAI Node.js SDK](https://github.com/openai/openai-node) · [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) · [Anthropic TypeScript SDK](https://github.com/anthropics/anthropic-sdk-typescript) · [Google Gemini API](https://ai.google.dev/gemini-api/docs) · [Vercel AI SDK](https://sdk.vercel.ai/) · [GitHub Copilot SDK](https://github.blog/news-insights/company-news/build-an-agent-into-any-app-with-the-github-copilot-sdk/) · [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) · [OpenRouter (Multi-Provider Gateway)](https://openrouter.ai/docs) |
| **Tokenization** | [OpenAI Tokenizer](https://platform.openai.com/tokenizer) · [tiktoken (js port)](https://github.com/dqbd/tiktoken) · [gpt-tokenizer](https://github.com/niieani/gpt-tokenizer) |
| **Extended Context & Caching** | [Prompt Caching (Anthropic)](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) · [Context Caching (OpenAI)](https://platform.openai.com/docs/guides/prompt-caching) · [Long Context Strategies](https://www.anthropic.com/research/many-shot-jailbreaking) · [Gemini 2M Context](https://ai.google.dev/gemini-api/docs/long-context) |
| **Local & Open-Source Models** | [Ollama](https://ollama.com/) · [Ollama.js](https://github.com/ollama/ollama-js) · [LM Studio](https://lmstudio.ai/) · [DeepSeek Models](https://www.deepseek.com/) · [Hugging Face Transformers.js](https://huggingface.co/docs/transformers.js) |
| **Courses** | [Vercel AI SDK Docs & Tutorials](https://sdk.vercel.ai/docs) · [AI Engineer Roadmap Video Series (dswh)](https://www.youtube.com/playlist?list=PLIkXejH7XPT8x9iUGvlsYt44aPx6ns8BV) |

### Projects to Build

1. **Prompt-driven CLI tool** — A command-line utility that takes a user query, calls an LLM API, and returns structured JSON output. Handle streaming, retries, and token counting.
2. **Prompt regression test suite** — Build a test harness that runs a battery of prompts against an LLM and asserts expected outputs using [Promptfoo](https://github.com/promptfoo/promptfoo). Track cost and latency per test.
3. **Multi-provider chatbot** — Build a Next.js conversational app using the [Vercel AI SDK](https://sdk.vercel.ai/) that lets the user switch between OpenAI, Anthropic, and a local Ollama model. Compare response quality, latency, and cost side-by-side.

---

## Level 2 — Single-Agent Systems

Build a single agent that can reason about a task, decide to use tools, and produce a final answer.

### Core Concepts

- [ ] The ReAct pattern: Reason → Act → Observe → Repeat
- [ ] Agent loop implementation (while not done: think → act → observe)
- [ ] Tool/function calling: defining schemas, parsing LLM tool calls, executing functions
- [ ] Parallel tool calling: executing multiple tools concurrently, handling dependencies
- [ ] Streaming tool calls: processing tool requests as they arrive, not waiting for full response
- [ ] Scratchpad / chain-of-thought as part of the agent state
- [ ] Stop conditions: max iterations, confidence thresholds, explicit "final answer"
- [ ] Error handling inside the loop (tool failures, malformed LLM output, timeouts)
- [ ] System prompt design for agent behavior (role, constraints, output format)
- [ ] Structured output parsing and validation (Zod, TypeBox, JSON Schema)

### Resources

| Topic | Links |
|---|---|
| **ReAct Pattern** | [ReAct: Synergizing Reasoning and Acting (Paper)](https://arxiv.org/abs/2210.03629) · [ReAct Prompting Explained](https://www.promptingguide.ai/techniques/react) |
| **Agent Fundamentals** | [LLM Powered Autonomous Agents (Lilian Weng)](https://lilianweng.github.io/posts/2023-06-23-agent/) · [Building Effective Agents (Anthropic)](https://www.anthropic.com/research/building-effective-agents) · [What Are AI Agents? (OpenAI)](https://openai.com/index/what-are-ai-agents/) · [GitHub Copilot Agentic Capabilities](https://github.blog/ai-and-ml/github-copilot/how-to-maximize-github-copilots-agentic-capabilities/) · [Functions, Tools and Agents with LangChain (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/) |
| **Function Calling** | [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) · [Anthropic Tool Use Docs](https://platform.claude.com/docs/en/build-with-claude/tool-use/overview) · [Claude Agent Skills Overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) |
| **Agent Frameworks** | TypeScript: [LangGraph.js](https://langchain-ai.github.io/langgraphjs/) · [Vercel AI SDK](https://sdk.vercel.ai/) · [OpenAI Agents SDK](https://platform.openai.com/docs/agents) · [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) · [Mastra](https://mastra.ai/) <br/> Python reference: [OpenAI Swarm](https://github.com/openai/swarm) · [CrewAI](https://www.crewai.com/) · [AutoGen](https://microsoft.github.io/autogen/) |
| **Testing Agents** | [Testing AI Applications](https://www.braintrust.dev/docs/guides/testing) · [LangSmith Testing](https://docs.smith.langchain.com/evaluation) · [MSW (Mock Service Worker)](https://mswjs.io/) |
| **Output Validation** | [Zod](https://zod.dev/) · [TypeBox](https://github.com/sinclairzx81/typebox) · [Instructor-JS](https://github.com/instructor-ai/instructor-js) |
| **Tracing** | [LangSmith Docs](https://docs.smith.langchain.com/) · [Arize Phoenix](https://github.com/Arize-ai/phoenix) · [Braintrust](https://www.braintrust.dev/) |

### Projects to Build

1. **Research agent from scratch** — Given a question, the agent searches the web (via API), reads results, decides if it has enough information, and synthesizes an answer with citations. Implement the full ReAct loop without a framework. Reference: [Building a ReAct Agent from Scratch](https://www.anthropic.com/research/building-effective-agents).
2. **Code review agent** — Takes a Git diff as input, analyzes it using an LLM, and produces structured feedback (issues, suggestions, severity). Use function calling to interact with a local file system.
3. **AI interview buddy** — Build an agent that conducts interactive mock interviews for a specific role. The agent asks questions, evaluates answers, provides feedback, and adapts difficulty based on performance.

---

## Level 3 — Tool Design & External Integrations

Agents are only as useful as the tools they can call. This level focuses on building reliable, production-grade tool interfaces.

### Core Concepts

- [ ] Tool definition: name, description, parameter JSON schema
- [ ] Writing tool descriptions that LLMs can select correctly
- [ ] Tool execution: sandboxing, timeouts, error propagation
- [ ] Authentication and authorization for tool calls (API keys, OAuth tokens, scopes)
- [ ] Idempotency and side effects — read vs. write tools
- [ ] Tool registries: static vs. dynamic tool lists
- [ ] Model Context Protocol (MCP) for standardized tool interfaces
- [ ] Rate limiting and circuit breakers on tool calls
- [ ] Returning tool results to the LLM: truncation, summarization, pagination

### Resources

| Topic | Links |
|---|---|
| **MCP & A2A** | [MCP Specification](https://modelcontextprotocol.io/) · [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) · [Building MCP Servers](https://platform.claude.com/docs/en/agents-and-tools/mcp) · [A2A Protocol](https://a2a.to/) |
| **Tool Design** | [OpenAI Function Calling Best Practices](https://platform.openai.com/docs/guides/function-calling#best-practices) · [Anthropic Tool Use Best Practices](https://platform.claude.com/docs/en/build-with-claude/tool-use/overview#best-practices) · [Claude Agent Skills Best Practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices) |
| **Sandboxing** | [E2B (Code Execution Sandbox)](https://e2b.dev/) · [gVisor](https://gvisor.dev/) · [Firecracker (MicroVMs)](https://firecracker-microvm.github.io/) · [WebAssembly (WASI)](https://wasi.dev/) |
| **Browser Automation** | [Playwright](https://playwright.dev/) · [Puppeteer](https://pptr.dev/) · [Browserbase](https://www.browserbase.com/) |
| **Circuit Breakers** | [Circuit Breaker Pattern (Martin Fowler)](https://martinfowler.com/bliki/CircuitBreaker.html) · [Cockatiel (Node.js Resilience)](https://github.com/connor4312/cockatiel) · [p-retry](https://github.com/sindresorhus/p-retry) |

### Projects to Build

1. **MCP-based tool server** — Build a Model Context Protocol server that exposes 3–5 tools (e.g., database query, file read/write, HTTP fetch). Follow the [MCP quickstart](https://modelcontextprotocol.io/quickstart/server). Handle auth, validation, and error responses.
2. **Agent with write access** — Build an agent that can create and update records in a database. Implement confirmation steps, dry-run mode, and rollback for destructive operations.

---

## Level 4 — Retrieval, Planning & Advanced Reasoning

Add higher-order capabilities: RAG retrieval, planning, self-correction, code execution, and multi-modal understanding.

### Core Concepts

- [ ] RAG (Retrieval-Augmented Generation): chunking, embedding, indexing, retrieval, reranking
- [ ] Vector databases: indexing strategies, similarity search, metadata filtering
- [ ] Hybrid search: combining vector similarity with keyword (BM25) search
- [ ] Code execution as a tool: sandboxed environments, capturing stdout/stderr
- [ ] Task decomposition: breaking complex tasks into sub-tasks
- [ ] Planning: generating a plan, executing steps, re-planning on failure
- [ ] Self-correction: detecting errors in own output, retrying with feedback
- [ ] Critique / reflection patterns: using a second LLM call to verify the first
- [ ] Fine-tuning: when it's the right tool (and when better prompting or RAG is sufficient)
- [ ] Multi-modal inputs: image, audio, and document understanding as agent capabilities

### Resources

| Topic | Links |
|---|---|
| **RAG** | [RAG From Scratch (LangChain series)](https://github.com/langchain-ai/rag-from-scratch) · [Retrieval-Augmented Generation (original paper)](https://arxiv.org/abs/2005.11401) · [Pinecone: What is RAG?](https://www.pinecone.io/learn/retrieval-augmented-generation/) |
| **Vector Databases** | [Pinecone](https://www.pinecone.io/) · [Weaviate](https://weaviate.io/) · [Qdrant](https://qdrant.tech/) · [pgvector](https://github.com/pgvector/pgvector) · [Chroma](https://www.trychroma.com/) |
| **Embeddings** | [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) · [MTEB Leaderboard (Model Comparisons)](https://huggingface.co/spaces/mteb/leaderboard) · [Transformers.js (Run Embeddings in Node)](https://huggingface.co/docs/transformers.js) |
| **Chunking Strategies** | [Chunking Strategies for LLM Applications (Pinecone)](https://www.pinecone.io/learn/chunking-strategies/) · [Five Levels of Chunking (Greg Kamradt)](https://youtu.be/8OJC21T2SL4) |
| **Planning & Reflection** | [Reflexion: Language Agents with Verbal Reinforcement (Paper)](https://arxiv.org/abs/2303.11366) · [Plan-and-Solve Prompting (Paper)](https://arxiv.org/abs/2305.04091) |
| **Code Execution** | [E2B Docs](https://e2b.dev/docs) · [Modal](https://modal.com/) |
| **Fine-Tuning** | [OpenAI Fine-Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning) · [When to Fine-Tune LLMs](https://www.anthropic.com/research/building-effective-agents) |
| **Multi-Modal** | [OpenAI Vision API](https://platform.openai.com/docs/guides/vision) · [Anthropic Vision Docs](https://platform.claude.com/docs/en/build-with-claude/vision) · [Vercel AI SDK: Multi-Modal](https://sdk.vercel.ai/docs/foundations/overview) |
| **Advanced RAG** | [Building and Evaluating Advanced RAG (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) · [Advanced RAG Cheat Sheet](https://github.com/NirDiamant/RAG_Techniques) · [GraphRAG (Microsoft)](https://microsoft.github.io/graphrag/) · [Evaluating RAG Components (Hugging Face Cookbook)](https://huggingface.co/learn/cookbook/en/rag_evaluation) |

### Projects to Build

1. **RAG agent over a codebase** — Index a real codebase into a vector store. Build an agent that answers questions about the code by retrieving relevant files. Compare naive RAG vs. [hybrid search](https://qdrant.tech/articles/hybrid-search/) vs. reranking with [Cohere Rerank](https://cohere.com/rerank).
2. **Self-correcting data pipeline agent** — Build an agent that writes and executes SQL queries to answer questions about a dataset. When a query fails, the agent analyzes the error and retries with a corrected query.
3. **Domain Q&A agent with fine-tuned model** — Fine-tune a small open-source model (e.g., Llama/Mistral with LoRA) on domain-specific data (legal, medical, financial). Build an agent that uses the fine-tuned model for domain reasoning and a larger model for general orchestration. Compare accuracy and cost vs. RAG-only.
4. **Multi-modal document agent** — Build an agent that processes PDFs with text, tables, and images. Use vision capabilities to understand charts/diagrams and text extraction for content. Answer questions requiring reasoning across both modalities.

---

## Level 5 — Multi-Agent Orchestration

Coordinate multiple specialized agents to handle complex workflows.

### Core Concepts

- [ ] Supervisor pattern: one agent delegates to specialized sub-agents
- [ ] Router / dispatcher: classify incoming tasks and route to the right agent
- [ ] DAG-based pipelines: define agent workflows as directed acyclic graphs
- [ ] Parallel fan-out: run multiple agents concurrently, aggregate results
- [ ] Agent-to-agent communication: message passing, shared state, handoffs
- [ ] Human-in-the-loop: approval gates, escalation, feedback injection
- [ ] Conflict resolution: when agents disagree or produce contradictory outputs
- [ ] Cost and latency budgets across multi-agent flows

### Resources

| Topic | Links |
|---|---|
| **Multi-Agent Patterns** | [Multi-Agent Systems (LangGraph.js)](https://langchain-ai.github.io/langgraphjs/concepts/multi_agent/) · [OpenAI Swarm Framework](https://github.com/openai/swarm) · [CrewAI: Multi-Agent Workflows](https://docs.crewai.com/) · [Mastra: Multi-Agent Workflows](https://mastra.ai/docs) · [AutoGen: Multi-Agent Conversation](https://microsoft.github.io/autogen/) |
| **Supervisor & Router** | [LangGraph.js Supervisor Tutorial](https://langchain-ai.github.io/langgraphjs/tutorials/multi_agent/agent_supervisor/) · [Build a Customer Support Bot (LangGraph.js)](https://langchain-ai.github.io/langgraphjs/tutorials/customer-support/customer-support/) |
| **DAG Pipelines** | [LangGraph.js: How-to Guides](https://langchain-ai.github.io/langgraphjs/how-tos/) · [LangGraph Cloud](https://langchain-ai.github.io/langgraph/cloud/) · [Inngest (Durable Functions)](https://www.inngest.com/) · [Temporal (TypeScript SDK)](https://docs.temporal.io/develop/typescript) |
| **Human-in-the-Loop** | [LangGraph.js: Human-in-the-Loop](https://langchain-ai.github.io/langgraphjs/concepts/human_in_the_loop/) · [Inngest: Wait for Event](https://www.inngest.com/docs/features/inngest-functions/steps-workflows/wait-for-event) |
| **Case Studies** | [How Replit Built Their Agent (Blog)](https://blog.replit.com/agent) · [Devin Architecture Analysis](https://www.cognition.ai/blog/introducing-devin) · [Cursor Agent Technical Overview](https://www.cursor.com/blog) |

### Projects to Build

1. **Multi-agent support system** — Build a system with a router agent that classifies incoming requests and delegates to specialized agents (billing, technical support, account management). Include a human escalation path. Follow the [LangGraph customer support tutorial](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/) as a starting point.
2. **Research → Draft → Review pipeline** — Build a three-agent pipeline: one researches a topic, one drafts a report, one reviews and provides feedback. Implement as a DAG with retry logic at each stage.
3. **E2E e-commerce product assistant** — Build a multi-agent system for an e-commerce domain: one agent handles product search (RAG over catalog), one handles order management (database tools), one handles returns/support. Coordinate via a supervisor with shared customer context.

---

## Level 6 — Memory & State

Give agents the ability to remember across conversations and share knowledge.

### Core Concepts

---

## Level 5 — Memory & State

Give agents the ability to remember across conversations and share knowledge.

### Core Concepts

- [ ] Short-term memory: conversation history within a single session
- [ ] Context window management: summarization, sliding window, token budgets
- [ ] Long-term memory: persisting facts, preferences, and past interactions
- [ ] Episodic vs. semantic memory: what happened vs. what is known
- [ ] Memory retrieval: when and how to pull relevant memories into context
- [ ] Shared state between agents: reading/writing to a common store
- [ ] Checkpointing: saving and restoring agent state mid-workflow
- [ ] Memory decay and eviction: preventing unbounded growth

### Resources

| Topic | Links |
|---|---|
| **Memory Concepts** | [LLM Powered Agents — Memory (Lilian Weng)](https://lilianweng.github.io/posts/2023-06-23-agent/#memory) · [LangGraph.js: Memory Concepts](https://langchain-ai.github.io/langgraphjs/concepts/memory/) |
| **Memory Systems** | [Mem0 (Memory Layer for AI)](https://github.com/mem0ai/mem0) · [Zep (Long-Term Memory)](https://www.getzep.com/) · [LangGraph.js Persistence](https://langchain-ai.github.io/langgraphjs/concepts/persistence/) |
| **Context Management** | [Managing Conversation History (OpenAI Cookbook)](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) · [Vercel AI SDK: Managing Conversations](https://sdk.vercel.ai/docs/foundations/overview) |
| **Checkpointing** | [LangGraph.js Checkpointers](https://langchain-ai.github.io/langgraphjs/concepts/persistence/#checkpoints) · [Temporal TypeScript SDK: Workflow State](https://docs.temporal.io/develop/typescript) |

### Projects to Build

1. **Stateful assistant with long-term memory** — Build a conversational agent that remembers user preferences, past interactions, and project context across sessions. Use [Mem0](https://github.com/mem0ai/mem0) or a custom PostgreSQL-based store. Implement storage, retrieval, and context window management.
2. **Collaborative agent workspace** — Build a system where two agents work on a shared document. Each can read and write to a shared state store. Implement conflict resolution and change tracking.

---

## Level 6 — Multi-Agent Orchestration

Coordinate multiple specialized agents to handle complex workflows.

### Core Concepts

- [ ] Supervisor pattern: one agent delegates to specialized sub-agents
- [ ] Router / dispatcher: classify incoming tasks and route to the right agent
- [ ] DAG-based pipelines: define agent workflows as directed acyclic graphs
- [ ] Parallel fan-out: run multiple agents concurrently, aggregate results
- [ ] Agent-to-agent communication: message passing, shared state, handoffs
- [ ] Human-in-the-loop: approval gates, escalation, feedback injection
- [ ] Conflict resolution: when agents disagree or produce contradictory outputs
- [ ] Cost and latency budgets across multi-agent flows

### Resources

| Topic | Links |
|---|---|
| **Multi-Agent Patterns** | [Multi-Agent Systems (LangGraph.js)](https://langchain-ai.github.io/langgraphjs/concepts/multi_agent/) · [OpenAI Swarm Framework](https://github.com/openai/swarm) · [CrewAI: Multi-Agent Workflows](https://docs.crewai.com/) · [Mastra: Multi-Agent Workflows](https://mastra.ai/docs) · [AutoGen: Multi-Agent Conversation](https://microsoft.github.io/autogen/) |
| **Supervisor & Router** | [LangGraph.js Supervisor Tutorial](https://langchain-ai.github.io/langgraphjs/tutorials/multi_agent/agent_supervisor/) · [Build a Customer Support Bot (LangGraph.js)](https://langchain-ai.github.io/langgraphjs/tutorials/customer-support/customer-support/) |
| **DAG Pipelines** | [LangGraph.js: How-to Guides](https://langchain-ai.github.io/langgraphjs/how-tos/) · [LangGraph Cloud](https://langchain-ai.github.io/langgraph/cloud/) · [Inngest (Durable Functions)](https://www.inngest.com/) · [Temporal (TypeScript SDK)](https://docs.temporal.io/develop/typescript) |
| **Human-in-the-Loop** | [LangGraph.js: Human-in-the-Loop](https://langchain-ai.github.io/langgraphjs/concepts/human_in_the_loop/) · [Inngest: Wait for Event](https://www.inngest.com/docs/features/inngest-functions/steps-workflows/wait-for-event) |
| **Case Studies** | [How Replit Built Their Agent](https://blog.replit.com/agent) · [Devin Architecture Analysis](https://www.cognition.ai/blog/introducing-devin) · [Cursor Agent Technical Overview](https://www.cursor.com/blog) |

### Projects to Build

1. **Multi-agent support system** — Build a system with a router agent that classifies incoming requests and delegates to specialized agents (billing, technical support, account management). Include a human escalation path.
2. **Research → Draft → Review pipeline** — Build a three-agent pipeline: one researches a topic, one drafts a report, one reviews and provides feedback. Implement as a DAG with retry logic at each stage.
3. **E2E e-commerce product assistant** — Build a multi-agent system for an e-commerce domain: one agent handles product search (RAG over catalog), one handles order management (database tools), one handles returns/support. Coordinate via a supervisor with shared customer context.

---

## Level 7 — Production & Operations

Ship agent systems that are reliable, observable, safe, and cost-efficient.

### Core Concepts

- [ ] Tracing: full trace of agent reasoning, tool calls, LLM inputs/outputs
- [ ] Evaluation: automated evals (correctness, relevance, safety), human evals
- [ ] Guardrails: input validation, output filtering, PII detection, content safety
- [ ] Cost tracking: per-request token usage, tool call costs, total cost per task
- [ ] Compute optimization: reasoning effort budgets, model routing, speculative decoding
- [ ] Latency optimization: prompt caching, model selection, parallel tool calls, streaming
- [ ] Deployment: containerization, autoscaling, queue-based architectures
- [ ] Failure modes: graceful degradation, fallback models, circuit breakers
- [ ] Security: prompt injection defense, tool authorization, sandboxing
- [ ] A/B testing: comparing agent versions, prompt variants, model versions
- [ ] Versioning: prompt versioning, tool schema versioning, agent behavior versioning
- [ ] LLMOps: CI/CD for prompts, automated eval in pipelines, model registry
- [ ] GPU inference optimization: batching, KV cache, quantization (for self-hosted models)
- [ ] Red teaming: systematic adversarial testing of agent behavior

### Resources

| Topic | Links |
|---|---|
| **Observability & Tracing** | [LangSmith Docs](https://docs.smith.langchain.com/) · [Langfuse (Open-Source LLM Engineering)](https://langfuse.com/) · [Arize Phoenix](https://github.com/Arize-ai/phoenix) · [Braintrust](https://www.braintrust.dev/) · [Helicone (LLM Observability)](https://www.helicone.ai/) · [OpenLIT](https://github.com/openlit/openlit) · [OpenTelemetry](https://opentelemetry.io/) |
| **Evaluation** | [Promptfoo (LLM Eval Framework)](https://github.com/promptfoo/promptfoo) · [Ragas (RAG Evaluation)](https://github.com/explodinggradients/ragas) · [DeepEval](https://github.com/confident-ai/deepeval) · [OpenAI Evals](https://github.com/openai/evals) |
| **Guardrails** | [Guardrails AI](https://www.guardrailsai.com/) · [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) · [Lakera Guard](https://www.lakera.ai/) |
| **Security** | [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) · [Prompt Injection Primer (Simon Willison)](https://simonwillison.net/2025/Apr/9/mcp-prompt-injection/) · [Rebuff (Prompt Injection Detection)](https://github.com/protectai/rebuff) |
| **Deployment** | [Docker](https://www.docker.com/) · [Kubernetes](https://kubernetes.io/) · [AWS Bedrock Agents](https://aws.amazon.com/bedrock/agents/) · [Vercel AI](https://vercel.com/ai) · [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/) · [Modal](https://modal.com/) · [Fly.io](https://fly.io/) · [Railway](https://railway.app/) |
| **Cost Optimization** | [OpenAI API Pricing](https://openai.com/api/pricing/) · [Anthropic Pricing](https://www.anthropic.com/pricing) · [Prompt Caching (Anthropic)](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) · [LLM Pricing Calculator](https://llmpricecheck.com/) |
| **LLMOps** | [LLMOps Guide (Databricks)](https://www.databricks.com/glossary/llmops) · [LLMOps (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/llmops/) · [Anthropic Workbench](https://docs.anthropic.com/en/docs/build-with-claude/workbench) · [Inngest (Durable Agent Workflows)](https://www.inngest.com/) · [Weights & Biases (Prompts)](https://wandb.ai/site/prompts) |
| **Red Teaming** | [Red Teaming LLM Applications (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/red-teaming-llm-applications/) · [Automated Testing for LLMOps (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/automated-testing-llmops/) · [AI Risk Resources](https://owasp.org/www-project-top-10-for-large-language-model-applications/) |

### Projects to Build

1. **Observable agent with full tracing** — Take an existing agent and add end-to-end tracing (every LLM call, tool call, decision point). Use [LangSmith](https://docs.smith.langchain.com/) or [Arize Phoenix](https://github.com/Arize-ai/phoenix). Build a dashboard showing latency, cost, and success rate per request.
2. **Eval suite for an agent** — Build an automated evaluation pipeline using [Promptfoo](https://github.com/promptfoo/promptfoo) that tests an agent against a dataset of inputs and expected outputs. Include metrics for correctness, latency, cost, and safety. Run it in CI.
3. **Hardened production agent** — Deploy an agent behind an API with: rate limiting, input guardrails, output filtering, graceful fallback to a simpler model on timeout, and cost budgets per user.

---

## Level 8 — Capstone Project

End-to-end integrative project that validates your mastery across all levels.

### Production Multi-Agent System

Build a complete production-ready multi-agent system that demonstrates:

1. **Agent router** that classifies requests and routes to appropriate specialized agents
2. **At least 3 specialized agents** using different tools (database, HTTP API, file operations)
3. **RAG retrieval** with vector store for knowledge augmentation
4. **Shared memory/state** across agents for context continuity
5. **Full observability** with tracing (LangSmith or Phoenix)
6. **Automated evaluation suite** in CI using Promptfoo
7. **Input/output guardrails** for safety and policy compliance
8. **Cost tracking** per request with budget management
9. **HTTP API deployment** with authentication and rate limiting
10. **Architecture documentation** including evaluation results and deployment strategy

---

## Recommended Learning Path

```
Week 1-2    → Level 0-1: Foundations + LLM Foundations
Week 3-4    → Level 2: Single-Agent Systems (ReAct loop, tool use)
Week 5-6    → Level 3: Tool Design & External Integrations (MCP, A2A, sandboxing)
Week 7-8    → Level 4: Retrieval, Planning & Advanced Reasoning (RAG, planning)
Week 9-10   → Level 5: Memory & State (long-term memory, checkpointing)
Week 11-12  → Level 6: Multi-Agent Orchestration (supervisors, DAGs)
Week 13-16  → Level 7: Production & Operations (tracing, evals, guardrails)
Week 17-20  → Level 8: Capstone Project (end-to-end production system)
```

---

## Key Architectural Patterns

| Pattern | Description | When to Use |
|---|---|---|
| **ReAct** | Reason → Act → Observe loop | Single-agent tasks requiring tool use |
| **Supervisor** | One agent delegates to specialized sub-agents | Complex tasks with distinct sub-domains |
| **Router** | Classify and dispatch to the right agent | Multi-tenant or multi-capability systems |
| **DAG Pipeline** | Chain agents in a directed graph | Sequential or parallel multi-step workflows |
| **Map-Reduce** | Fan out to many agents, aggregate results | Parallel processing of independent sub-tasks |
| **Critic / Reflection** | A second pass reviews the first | Tasks requiring high accuracy or safety |
| **Human-in-the-Loop** | Agent pauses for human approval | Write operations, high-stakes decisions |
| **Fallback Chain** | Try stronger model first, fall back on failure/timeout | Balancing quality, latency, and cost |

---

## Glossary

| Term | Definition |
|---|---|
| **Agent** | A system that uses an LLM to decide which actions to take in a loop until a task is completed. |
| **ReAct** | A prompting pattern where the model alternates between reasoning (thinking) and acting (calling tools). |
| **Tool / Function Calling** | The mechanism by which an LLM outputs a structured request to invoke an external function. |
| **MCP (Model Context Protocol)** | An open standard for connecting LLM agents to external tools and data sources through a unified interface. |
| **A2A (Agent-to-Agent) Protocol** | An open standard for inter-agent communication across service boundaries, complementing MCP's agent-to-tool focus. |
| **RAG (Retrieval-Augmented Generation)** | A pattern where relevant documents are retrieved from a store and injected into the LLM's context before generation. |
| **Vector Store** | A database optimized for storing and querying high-dimensional embeddings for similarity search. |
| **Embedding** | A fixed-length numerical vector representing the semantic meaning of a piece of text. |
| **Context Window** | The maximum number of tokens (input + output) an LLM can process in a single request. |
| **Token Budget** | An allocated limit on the number of tokens an agent or workflow can consume, used to control costs in production systems. |
| **Guardrails** | Input/output validation layers that enforce safety, format, and policy constraints on agent behavior. |
| **Checkpointing** | Saving the state of an agent or workflow at a point in time so it can be resumed or replayed. |
| **Orchestrator** | A component that coordinates multiple agents, managing task routing, state, and control flow. |
| **Prompt Injection** | An attack where adversarial input manipulates the LLM into ignoring its instructions or performing unintended actions. |
| **Structured Output** | Constraining an LLM to output valid JSON, XML, or another machine-parseable format. |
| **Token** | The basic unit of text processed by an LLM — roughly ¾ of a word in English. |
| **Reasoning Tokens** | Hidden chain-of-thought tokens generated by reasoning models (o1, R1) before producing the final answer. Controllable via max reasoning effort parameters. |
| **Eval** | An automated test that measures agent quality (correctness, relevance, safety, cost, latency). |

---

## Related Roadmaps

Other roadmaps worth exploring alongside this one:

| Roadmap | Focus | Link |
|---|---|---|
| **DotNet Developer Roadmap** | Inspiration for this roadmap's structure and format | [milanm/DotNet-Developer-Roadmap](https://github.com/milanm/DotNet-Developer-Roadmap) |
| **AI Engineer Roadmap** | Beginner→Advanced AI engineering path with course links | [dswh/ai-engineer-roadmap](https://github.com/dswh/ai-engineer-roadmap) |
| **AI Expert Roadmap** | Broad AI/ML/Data Science landscape with interactive charts | [AMAI-GmbH/AI-Expert-Roadmap](https://github.com/AMAI-GmbH/AI-Expert-Roadmap) |
| **Complete Roadmap to Learn AI** | Three-path approach: Data Science, GenAI, Agentic AI | [krishnaik06/Complete-RoadMap-To-Learn-AI](https://github.com/krishnaik06/Complete-RoadMap-To-Learn-AI) |
| **Agentic AI Roadmap (Krish Naik)** | Complementary agentic AI learning path | [krishnaik06/Roadmap-To-Learn-Agentic-AI](https://github.com/krishnaik06/Roadmap-To-Learn-Agentic-AI) |
| **Prompt Engineering Guide** | Deep dive into prompting techniques | [dair-ai/Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide) |
| **RAG Techniques** | Comprehensive collection of RAG patterns | [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) |

---

## License

[Apache License 2.0](LICENSE)
