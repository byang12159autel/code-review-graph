<p align="center">
  <img src="docs/assets/banner.jpg" alt="code-review-graph" width="100%" />
</p>

<h1 align="center">code-review-graph</h1>

<p align="center">
  <a href="https://github.com/tirth8205/code-review-graph/stargazers"><img src="https://img.shields.io/github/stars/tirth8205/code-review-graph?style=flat-square" alt="Stars"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square" alt="MIT Licence"></a>
  <a href="https://github.com/tirth8205/code-review-graph/actions/workflows/ci.yml"><img src="https://github.com/tirth8205/code-review-graph/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square" alt="Python 3.10+"></a>
  <a href="https://modelcontextprotocol.io/"><img src="https://img.shields.io/badge/MCP-compatible-green.svg?style=flat-square" alt="MCP"></a>
  <a href="#"><img src="https://img.shields.io/badge/version-1.7.2-purple.svg?style=flat-square" alt="v1.7.2"></a>
</p>

Claude Code re-reads your entire codebase on every task. `code-review-graph` fixes that. It builds a structural map of your code with [Tree-sitter](https://tree-sitter.github.io/tree-sitter/), tracks changes incrementally, and gives Claude precise context so it reads only what matters.

Benchmarked on three production open-source projects: **6.8x fewer tokens for code reviews, up to 49x fewer tokens for coding tasks.**

<p align="center">
  <img src="docs/assets/marketing-diagram.png" alt="code-review-graph architecture overview" width="100%" />
</p>

## Installation

**Claude Code Plugin** (recommended)

```bash
claude plugin add tirth8205/code-review-graph
```

**pip**

```bash
pip install code-review-graph
code-review-graph install
```

Restart Claude Code after either method. Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

## Getting Started

Open your project in Claude Code and run:

```
Build the code review graph for this project
```

The initial build takes roughly 10 seconds for a 500-file project. After that, the graph updates automatically on every file edit and git commit.

## How It Works

The graph maps every function, class, import, call, inheritance relationship, and test in your codebase. When you ask Claude to review code or make changes, it queries the graph first to determine what changed and what depends on those changes, then reads only the relevant files along with their blast-radius information rather than scanning everything.

You continue using Claude Code exactly as before. The graph operates in the background, updating itself as you work.

## Benchmarks

All figures come from real tests on three production open-source repositories.

### Code Reviews: 6.8x Token Reduction

Tested across 6 real git commits. The graph replaces reading entire source files with a compact structural summary (156 to 207 tokens) covering blast radius, test coverage gaps, and dependency chains.

| Repo | Size | Standard Approach | With Graph | Reduction | Review Quality |
|------|-----:|------------------:|-----------:|----------:|:-:|
| [httpx](https://github.com/encode/httpx) | 125 files | 12,507 tokens | 458 tokens | 26.2x | 9.0 vs 7.0 |
| [FastAPI](https://github.com/fastapi/fastapi) | 2,915 files | 5,495 tokens | 871 tokens | 8.1x | 8.5 vs 7.5 |
| [Next.js](https://github.com/vercel/next.js) | 27,732 files | 21,614 tokens | 4,457 tokens | 6.0x | 9.0 vs 7.0 |
| **Average** | | **13,205** | **1,928** | **6.8x** | **8.8 vs 7.2** |

Standard approach: reading all changed files plus the diff. Quality scored on accuracy, completeness, bug-catching potential, and actionable insight (1 to 10 scale).

### Live Coding Tasks: 14.1x Average, 49x Peak

An agent performed 6 real coding tasks (adding features, fixing bugs) across the same repositories. The graph directed it to the right files and away from everything else.

| Task | Repo | With Graph | Without Graph | Reduction | Files Skipped |
|------|------|--------:|-----------:|----------:|---:|
| Add rate limiter | httpx | 14,090 | 64,666 | 4.6x | 58 |
| Fix streaming bug | httpx | 14,090 | 64,666 | 4.6x | 59 |
| Add rate limiter | FastAPI | 37,217 | 138,585 | 3.7x | 1,120 |
| Fix streaming bug | FastAPI | 36,986 | 138,585 | 3.7x | 1,121 |
| Add rate limiter | Next.js | 15,049 | 739,352 | 49.1x | ~16,000 |
| Fix streaming bug | Next.js | 16,135 | 739,352 | 45.8x | ~16,000 |

The graph identified the correct files in every case. Savings scale with repository size: a 125-file project sees roughly 4.6x reduction, whilst a 27,000-file monorepo sees close to 49x.

## Usage

### Slash Commands

| Command | Description |
|---------|-------------|
| `/code-review-graph:build-graph` | Build or rebuild the code graph |
| `/code-review-graph:review-delta` | Review changes since last commit |
| `/code-review-graph:review-pr` | Full PR review with blast-radius analysis |

### CLI

```bash
code-review-graph install     # Register MCP server with Claude Code
code-review-graph build       # Parse entire codebase
code-review-graph update      # Incremental update (changed files only)
code-review-graph status      # Graph statistics
code-review-graph watch       # Auto-update on file changes
code-review-graph visualize   # Generate interactive HTML graph
code-review-graph serve       # Start MCP server
```

### MCP Tools

Claude uses these automatically once the graph is built.

| Tool | Description |
|------|-------------|
| `build_or_update_graph_tool` | Build or incrementally update the graph |
| `get_impact_radius_tool` | Blast radius of changed files |
| `get_review_context_tool` | Token-optimised review context with structural summary |
| `query_graph_tool` | Callers, callees, tests, imports, inheritance queries |
| `semantic_search_nodes_tool` | Search code entities by name or meaning |
| `embed_graph_tool` | Compute vector embeddings for semantic search |
| `list_graph_stats_tool` | Graph size and health |
| `get_docs_section_tool` | Retrieve documentation sections |

## Features

| Feature | Details |
|---------|---------|
| Incremental updates | Re-parses only changed files. Subsequent updates complete in under 2 seconds. |
| 12 languages | Python, TypeScript, JavaScript, Go, Rust, Java, C#, Ruby, Kotlin, Swift, PHP, C/C++ |
| Blast-radius analysis | Shows exactly which functions, classes, and files are affected by any change |
| Auto-update hooks | Graph updates on every file edit and git commit without manual intervention |
| Semantic search | Optional vector embeddings via sentence-transformers |
| Interactive visualisation | D3.js force-directed graph with edge-type toggles and search |
| Local storage | SQLite file in `.code-review-graph/`. No external database, no cloud dependency. |
| Watch mode | Continuous graph updates as you work |

## Configuration

To exclude paths from indexing, create a `.code-review-graphignore` file in your repository root:

```
generated/**
*.generated.ts
vendor/**
node_modules/**
```

For semantic search, install the optional embeddings dependencies:

```bash
pip install code-review-graph[embeddings]
```

## Contributing

```bash
git clone https://github.com/tirth8205/code-review-graph.git
cd code-review-graph
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

To add a new language, edit `code_review_graph/parser.py` and add your extension to `EXTENSION_TO_LANGUAGE` along with node type mappings in `_CLASS_TYPES`, `_FUNCTION_TYPES`, `_IMPORT_TYPES`, and `_CALL_TYPES`. Include a test fixture and open a PR.

## Licence

MIT. See [LICENSE](LICENSE).

<p align="center">
<br>
<code>pip install code-review-graph && code-review-graph install</code>
</p>
