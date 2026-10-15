# Deploy guide: Groq AI + a free public URL

Everything below is free. Two honest notes first:
* **There is no free `.com`.** Domain registration costs money (about $10-15/year). Free hosting gives you a free
  subdomain such as `drone-vision-ai.onrender.com`. You can attach a paid `.com` later (section 5).
* Free hosts are **demo-grade**: small CPU/RAM, they sleep when idle, and their disk is wiped on restart. This project is
  built for that (see "Demo mode"), but do not process large videos on a free instance.

## 1. Get a free Groq API key (about 2 minutes)
1. Go to https://console.groq.com, sign up, open **API Keys**, click **Create API Key**, copy `gsk_...`.
2. Put it in `.env` (Docker) or in the host's environment settings: `GROQ_API_KEY=gsk_...`
3. Restart. **Command Center -> AI engines** should now say `Assistant: llm+rules-fallback` and `Vision: groq`.

What Groq powers: the **assistant** (tool-calling over your incident database, answers in English/Hindi/Hinglish), the
**shift briefing**, **Explain this incident**, and **evidence-frame vision analysis**. Every answer is grounded: incident
codes / object ids the model mentions must exist in the tool results, otherwise the answer is rejected and the built-in rules
engine answers instead. If Groq is down, rate-limited or has no key, the app keeps working on the rules engine.

**Model IDs change.** Groq retires models often (checked Oct 2026: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant` and
`llama-4-scout` were already retired). Defaults are `GROQ_MODELS=openai/gpt-oss-120b,openai/gpt-oss-20b` (tool calling) and
`GROQ_VISION_MODELS=qwen/qwen3.8-27b` (vision). If you see `model_not_found`, look up current IDs at
https://console.groq.com/docs/models and set the env var (comma list = fallback chain). The last error is shown in the AI engines card.

Free-tier limits are per model and can change. The app protects the quota: `LLM_USER_PER_HOUR` (default 30) and `LLM_DAILY_CAP` (400).

## 2. Run locally with Docker
```
cp .env.example .env      # set SEED_PASSWORD, JWT_SECRET, GROQ_API_KEY
docker compose up --build # http://localhost:8080
```
Offline development without any key: `python scripts/mock_groq.py` and set
`GROQ_BASE_URL=http://127.0.0.1:9000/openai/v1`, `GROQ_API_KEY=mock` (a keyword-based stand-in, **not** an LLM).

## 3. Free public URL
### Option A - Render (recommended: connects to GitHub, auto-deploys on push)
1. Push this repo to GitHub.
2. https://render.com -> sign up with GitHub -> **New -> Blueprint** -> pick the repo -> **Apply**. (`render.yaml` is included.)
3. In the service's **Environment** tab set `GROQ_API_KEY` (and `PUBLIC_URL` to your onrender URL). `SEED_PASSWORD` and `JWT_SECRET`
   are generated for you; read `SEED_PASSWORD` there to sign in as `admin`.
4. First build takes several minutes. Then open `https://<service-name>.onrender.com`.

Free-tier facts to expect (verify on Render's pricing page, they change): 512 MB RAM, 0.1 CPU, sleeps after ~15 min idle
(first request after that takes up to a minute), ephemeral disk. A card may be requested at signup (sources disagree).
WebSockets (live view) work.

### Option B - your own PC, public in one command (no account, no card)
Full speed (your CPU), the link changes every run, and it only works while your PC and Docker are on:
```
winget install --id Cloudflare.cloudflared
docker compose up -d
cloudflared tunnel --url http://localhost:8080
```
It prints a `https://<random>.trycloudflare.com` link. Good for interviews. Use `DEMO_MODE=1` and a strong `SEED_PASSWORD`, and stop the tunnel afterwards.

### Option C - others
Hugging Face Docker Spaces, SnapDeploy, Fly/Koyeb/Railway: free-tier terms changed recently (some now need a card or a paid plan).
Check their current pricing before choosing; the Dockerfile honours `$PORT`, so any Docker host works.

## 4. Demo mode (`DEMO_MODE=1`)
* A sample mission with a pre-processed CRITICAL incident is **baked into the image** (so it survives free-tier restarts).
* The sign-in page shows **"Try the live demo"**: a read-only `demo` account (role *analyst*): it can browse, ask the assistant, generate briefings
  and reports (10/hour), but cannot upload, process, edit zones or manage users.
* Visitors' AI usage is rate-limited so a public link cannot burn your free Groq quota.

## 5. Your own `.com`
Buy a domain (Cloudflare Registrar sells at cost; Namecheap/GoDaddy also work, roughly $10-15/year). In Render:
**Settings -> Custom Domains -> Add**, create the CNAME record Render shows at your registrar, wait for TLS to issue.

## 6. Telegram / Slack / Discord alerts (optional)
HIGH/CRITICAL incidents can notify you the moment processing finishes.
* Telegram: talk to **@BotFather** -> `/newbot` -> token. Send your bot a message, then open
  `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `chat.id`. Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
* Slack/Discord: create an incoming webhook and set `ALERT_WEBHOOK_URL`.
* `ALERT_MIN_SEVERITY` (default `HIGH`), `PUBLIC_URL` (link in the message).

## 7. Security checklist before sharing a link
* `DEMO_MODE=1`, and never upload real/private footage to a public instance.
* Strong `SEED_PASSWORD` and a 32+ char `JWT_SECRET`; secrets only in the host's env settings, never in git.
* The admin account is not shown anywhere; only the read-only demo account is public.
* HTTPS is provided by Render / Cloudflare; the login token lives in `sessionStorage`.
