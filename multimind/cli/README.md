# MultiMindSDK CLI Tools

This directory contains the main command-line tools for the MultiMindSDK. Each CLI is modular, developer-friendly, and focused on a specific feature or workflow.

## Table of Contents
- [Agentic Workflow CLI (`agentic.py`)](#agentic-workflow-cli-agenticpy)
- [Feature-Specific CLIs](#feature-specific-clis)
- [How to Use](#how-to-use)
- [Entry Point](#entry-point)

---

## Agentic Workflow CLI (`agentic.py`)

**Run agentic, evolutionary, and reflexive workflow demos from the command line.**

### Usage
```bash
python -m multimind.cli.agentic list-examples
python -m multimind.cli.agentic run-demo reflexive
python -m multimind.cli.agentic run-demo hybrid
python -m multimind.cli.agentic info
```

### Commands
| Command         | Description                                      |
|----------------|--------------------------------------------------|
| list-examples   | List all available agentic workflow demos        |
| run-demo <name> | Run a specific demo (e.g., reflexive, hybrid)    |
| info            | Show CLI info and available examples             |

---

## Feature-Specific CLIs

| File                    | Purpose                                      |
|-------------------------|----------------------------------------------|
| context_transfer.py     | CLI for context transfer workflows           |
| model_conversion_cli.py | CLI for model conversion and export          |
| multi_model_cli.py      | CLI for multi-model management               |
| compliance.py           | CLI for compliance workflows                 |
| chat.py                 | CLI for chat and conversational agents       |
| config.py               | CLI for configuration management             |
| models.py               | CLI for model registry and management        |
| __main__.py             | Main entry point (see below)                 |

Each CLI is self-contained and can be run as a module:
```bash
python -m multimind.cli.<tool_name> [args]
```

---

## How to Use

- **List all available agentic workflow demos:**
  ```bash
  python -m multimind.cli.agentic list-examples
  ```
- **Run a specific agentic workflow demo:**
  ```bash
  python -m multimind.cli.agentic run-demo reflexive
  ```
- **Run a feature-specific CLI:**
  ```bash
  python -m multimind.cli.model_conversion_cli --help
  ```

---

## Entry Point

The file `__main__.py` allows you to run the CLI package directly:
```bash
python -m multimind.cli
```
This should dispatch to a main menu or point you to the right CLI tools.

---

## Developer Notes
- All CLI tools are modular and can be extended independently.
- For more examples and advanced usage, see the `examples/cli/` and `examples/evolutionary/` folders.
- Each CLI tool provides `--help` for command-line usage details.

---

Happy hacking! 🚀 