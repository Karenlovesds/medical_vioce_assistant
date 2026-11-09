# Medical Voice Assistant

A safety-first inbound phone agent for general medical education. It uses
LiveKit Agents with a classic STT -> LLM -> TTS pipeline and deterministic
safety rules.

## What it does

- Answers general medical education questions.
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
- `knowledge.py` - optional BM25 document search utility (not currently wired
  into the agent).
- `evaluation.py` - safety and guard-latency evaluation harness.
- `Dockerfile`, `.dockerignore` - deployment image.
- `requirements.txt` - pinned dependencies.

## Setup

```bash
cd medical_vioce_assistant
pip install -r requirements.txt
```

Create a `.env` with:

```bash
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
TRANSFER_NUMBER=+1XXXXXXXXXX
```

`.env` is ignored by git and excluded from Docker. Never commit it.

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
