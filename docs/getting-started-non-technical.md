# Getting Started Without Writing Code

A guide for analysts, compliance officers, project managers, and anyone else who wants to see what MultiMind does without being a programmer. No Python knowledge assumed. Every command in this guide can be copied and pasted exactly as written.

## What is MultiMind, in plain language

When a company sends text to an AI service (ChatGPT, Claude, and similar), three things can go wrong:

1. **Sensitive data leaks.** Someone pastes a customer's email address, social security number, or credit card into a prompt, and it leaves the building.
2. **Nobody can prove what happened.** If a regulator or auditor asks "did personal data ever get sent to the AI vendor?", there is no record.
3. **Costs run away.** Every AI call costs money, and without a meter nobody notices until the invoice arrives.

MultiMind is **seatbelts for AI**. It sits between your company's software and the AI service and provides:

- a **guard** — it spots sensitive data (emails, phone numbers, ID numbers, credit cards) and masks it *before* it is sent out, or blocks the request entirely;
- an **audit trail** — a tamper-evident log file recording what was caught and when, without ever storing the sensitive values themselves;
- a **budget** — a spending meter and a hard ceiling, so AI costs cannot silently run away.

It also lets developers swap between AI vendors easily, and can run AI models entirely on your own computer so that nothing is sent to any vendor at all. The rest of this guide lets you try the guard, the audit trail, and the local AI yourself.

## Step 0: Open a terminal

The terminal is a window where you type commands instead of clicking buttons.

- **Mac**: press Cmd+Space, type `Terminal`, press Enter.
- **Windows**: press the Windows key, type `PowerShell`, press Enter.

You type (or paste) a command at the prompt and press Enter to run it. That is the only skill this guide requires.

## Step 1: Install Python

Python is the language MultiMind is written in. Check whether you already have it — paste this into the terminal and press Enter:

```
python3 --version
```

If you see something like `Python 3.11.6` (any number from 3.9 up), you are done with this step. If you get an error instead, download Python from [python.org/downloads](https://www.python.org/downloads/) and run the installer, accepting the defaults. On Windows, tick the box that says "Add Python to PATH" during installation, then close and reopen the terminal.

## Step 2: Install MultiMind

Paste this and press Enter:

```
python3 -m pip install multimind-sdk
```

Lines of progress text will scroll by for a minute; that is normal. When the prompt comes back, verify it worked:

```
multimind --help
```

You should see a short list of available commands (`chat`, `compliance`, `config`, `models`). If `multimind` is reported as "command not found", close and reopen the terminal and try again.

## Step 3: Scan a file for sensitive data

This is the guard, usable directly from the terminal — no AI service, no API key, no internet connection needed. First try it on a sentence:

```
multimind compliance scan-text "Contact Jane at jane.doe@corp.com or 555-123-4567"
```

You will see a table like this:

```
                PII Findings (2)
┏━━━━━━━┳━━━━━━━┳━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Type  ┃ Start ┃ End ┃ Match             ┃
┡━━━━━━━╇━━━━━━━╇━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ email │    16 │  34 │ jane.doe@corp.com │
│ phone │    38 │  50 │ 555-123-4567      │
└───────┴───────┴─────┴───────────────────┘
```

How to read it:

- **Type** — what kind of sensitive data was found (email, phone, ssn, credit_card, iban, api_key, and more). "PII" stands for *personally identifiable information*.
- **Start / End** — the character positions in the text, so the finding can be located exactly.
- **Match** — the actual text that was flagged.

If nothing is found, it prints `No PII found.` The command also signals the result invisibly: it finishes with status "failure" when PII is found, which lets automated systems (for example, a check that runs before every document upload) refuse to proceed.

To scan a whole file — say a document called `customer_notes.txt` in your Downloads folder — and also see what a cleaned-up version would look like:

```
multimind compliance scan-text --file ~/Downloads/customer_notes.txt --redact mask
```

The `--redact mask` part prints the text again with every finding replaced by a label like `[EMAIL]` or `[CREDIT_CARD]` — this is exactly what the guard sends to the AI vendor instead of the real values.

## Step 4: Understand the audit trail

When developers wire the guard into an application, every event is appended to an audit file (usually named something like `compliance_audit.jsonl`). Each line is one event. A real line looks like this:

```json
{"timestamp": "2026-07-06T12:44:37+00:00", "direction": "input", "method": "generate",
 "pii_types": {"email": 1, "ssn": 1}, "count": 2, "strategy": "mask",
 "blocked": false, "tags": ["[EMAIL:86e0b9e5]", "[SSN:01a54629]"]}
```

Field by field:

- **timestamp** — when it happened, in universal time.
- **direction** — `input` means data on its way *to* the AI vendor; `output` means data coming *back*.
- **pii_types / count** — what was found and how many of each. Here: one email and one social security number.
- **strategy** — what was done about it. `mask` means the values were replaced with labels before sending.
- **blocked** — `true` means the request was refused entirely (used for the most sensitive types, like credit cards).
- **tags** — a short fingerprint (a hash) of each value. The same email always produces the same fingerprint, so an auditor can tell "this is the same person's email appearing in 14 requests" — without the file ever containing the email itself.

The important property for compliance: **the audit file never contains the sensitive values themselves** — only types, counts, and fingerprints. It can be handed to an auditor as-is.

## Step 5: Run a private AI on your own computer

This step runs an AI model entirely on your machine. Nothing you type is sent to any company. It uses Docker, a tool that runs pre-packaged software.

1. Install **Docker Desktop** from [docker.com](https://www.docker.com/products/docker-desktop/) and open it once so it is running (you will see a whale icon).

2. Download MultiMind's stack definition and start it. Paste these four commands one at a time:

```
git clone https://github.com/multimindlab/multimind-sdk.git
```

```
cd multimind-sdk
```

```
DEFAULT_MODEL=ollama docker compose --profile local up -d
```

```
docker compose exec ollama ollama pull mistral
```

What just happened, line by line: you downloaded the project; moved into its folder; started two services (the MultiMind gateway, plus Ollama — a program that runs AI models locally); and downloaded a free AI model called Mistral (about 4 GB — the last command takes a few minutes).

3. Talk to it:

```
curl -X POST http://localhost:8000/v1/chat -H 'Content-Type: application/json' -d '{"model": "ollama", "messages": [{"role": "user", "content": "Explain GDPR in two sentences"}]}'
```

The reply comes back as structured text (JSON). Look for the `"content"` part — that is the AI's answer. `localhost` means "this computer": the request never left your machine.

4. Prefer clicking to typing? Open http://localhost:8000/docs in your web browser. That page (called Swagger) lists everything the service can do and lets you try each function with a "Try it out" button.

5. When you are done, shut everything down with:

```
docker compose down
```

## What you have seen

- The **guard** catching sensitive data in text and files, with no internet needed (Step 3).
- The **audit trail** that proves, without exposing anything, what was caught (Step 4).
- A complete AI service running **privately on your own computer** (Step 5).

The third seatbelt — the **budget** — is wired in by developers; it meters every AI call in dollars and blocks calls once a spending ceiling is reached.

## Where to go next

- Hand your developers the [quickstart](quickstart.md) — the same features, wired into code, in five minutes.
- The [compliance guide](compliance.md) covers the full GDPR/HIPAA compliance module.
- Questions? Join the [Discord community](https://discord.gg/K64U65je7h) — non-technical questions are welcome.
