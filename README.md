# Medical Voice Assistant

A safety-first inbound phone agent for general diabetes education. It uses
LiveKit Agents with an OpenAI realtime model and deterministic safety rules.

## What it does

- Answers general diabetes education questions.
- Refuses diagnosis, dosing, insulin, and medication-change advice.
- Advises the caller to call 911 immediately and offers a human transfer;
  self-harm gets the 988 script.
- Offers a human transfer when the caller asks for a person.
- Hands off after repeated confusion.
- Gives the current date and time.

It does not provide clinical care, and it is not a HIPAA-compliant system.

## Files

- `agent.py` - agent entrypoint, prompt, tools, safety scripts, phone transfer.
- `safety.py` - deterministic emergency, self-harm, refusal, injection, and
  handoff checks.
- `knowledge.py` - BM25 index build and internal document search.
- `evaluation.py` - safety and guard-latency evaluation harness.
- `Dockerfile`, `.dockerignore` - deployment image.
- `requirements.txt` - pinned dependencies.

## Setup

```bash
cd medical_vioce_assistant
pip install -r requirements.txt
```

Put source documents in `data/docs/`, then build the search index:

```bash
python knowledge.py
```

PDF, `.txt`, and `.md` inputs are supported. Scanned PDFs need OCR before they
can be indexed.

Create a `.env` with:

```bash
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
TRANSFER_NUMBER=+1XXXXXXXXXX
```

`.env` is ignored by git and excluded from Docker. Never commit it.

## Internal documents

Source files live in `data/docs/` and are ignored by git and Docker. The
generated index is written to `data/index/chunks.json` and is ignored by git
but included in the Docker build context.

`knowledge.py` uses BM25 lexical search, so no embedding API key or external
service is needed. The agent calls `search_diabetes_docs` before answering a
diabetes education question, answers only from the returned passages, and cites
the source and page. If nothing relevant is found, it says it does not have that
information and suggests asking for a person.

Rebuild the index whenever documents change. If source documents are private,
build the index locally before `lk agent deploy`; the source files are not
uploaded.

## Run

```bash
python agent.py console --text   # text-only local test
python agent.py console          # local audio test
python agent.py dev              # connect to LiveKit
python agent.py start            # production mode
```

Inbound calls dispatch to the agent named `medical-agent`.

## Test

```bash
python evaluation.py
```

This checks safety actions, negation/education handling, confusion handoff, and
guard latency. It does not score LLM answer quality.

## Deploy

```bash
python knowledge.py   # build data/index/chunks.json first
lk agent create       # first time
lk agent deploy
lk agent status
lk agent logs
```

Set `TRANSFER_NUMBER` in the LiveKit Cloud agent environment; `.env` is not
uploaded. Transfer uses a cold SIP REFER, requires provider support, and does
not work in text console mode.

## Limitations

- No diagnosis, dosing, or medication changes.
- RAG is BM25 lexical retrieval only; it does not handle synonyms or paraphrases
  as well as semantic embeddings.
- No PHI program, BAA, or full de-identification.
- Regex safety checks are intentionally narrow; they are not full NLU.
- Accent, noise, and interruption coverage is manual.
