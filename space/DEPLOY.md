# Deploy the live demo (Hugging Face Space)

This folder is a self-contained Gradio Space. It installs `ask-santosh` from
GitHub, builds the index in memory on first request, and serves a chat UI.

## Option A — one command (recommended)

```bash
pip install huggingface_hub
export HF_TOKEN=hf_xxx            # write token: https://huggingface.co/settings/tokens
export OPENAI_API_KEY=sk-xxx      # optional: also stored as the Space secret
python space/deploy.py <your-hf-username>
```

That creates the Space, uploads `app.py` + `requirements.txt` + `README.md`, and
sets the secret. The live URL (`https://<user>-ask-santosh.hf.space`) appears on
the Space page once it finishes building.

## Option B — the web UI

1. Create a Space at <https://huggingface.co/new-space> → SDK **Gradio**.
2. Upload `app.py`, `requirements.txt`, and `README.md` from this folder.
3. In **Settings › Secrets**, add `OPENAI_API_KEY` (or `OPENAI_BASE_URL` +
   any OpenAI-compatible endpoint via `OPENAI_BASE_URL`).
4. Wait for the build; open the app.

## After deploying

Add the live URL to the profile's "Proof, Not Claims" table and to the repo
README badge row, e.g. `[Live demo →](https://<user>-ask-santosh.hf.space)`.

## Notes

- First request triggers ingestion (embeds the feed) and is slower; later
  requests are fast.
- Guardrails in `app.py`: 500-char question cap and a Gradio concurrency/queue
  limit. Consider Space-level rate limits before sharing widely.
