# Deploy: nepalgov.datacortex.in via Render

Runbook for the hosted demo on Render free tier with custom domain via
Hostinger DNS. This replaces the older Caddy/systemd runbook, which assumed
VPS + SSH access.

## Architecture

```
Browser → nepalgov.datacortex.in
       → Cloudflare/Hostinger DNS (CNAME)
       → Render edge (Render-issued TLS cert)
       → FastAPI app on Render free instance
       → loads RAG corpus once at startup
```

## One-time setup — about 25 minutes total

### Step 1 — Push the deploy files to the repo (5 min)

The repo already has `demo/app.py` (Gradio); we're replacing it with the
custom FastAPI frontend under `web/`. Three new things in the repo:

```
web/app.py                  # FastAPI backend
web/templates/index.html    # editorial landing page
web/static/style.css        # all styling
web/static/app.js           # frontend logic
render.yaml                 # Render blueprint
```

Commit and push these to `main`. No version bump required — this is
deploy plumbing, not a code release.

```bash
git add web/ render.yaml
git commit -m "Deploy: FastAPI frontend + Render blueprint for nepalgov.datacortex.in"
git push
```

### Step 2 — Create the Render service (5 min)

1. Sign in to https://dashboard.render.com
2. Click **"New +"** → **"Blueprint"**
3. Connect the GitHub repo `irfanalidv/Nepal-Gov-Agent`
4. Render reads `render.yaml` and shows the service it will create
5. Click **"Apply"**
6. First build takes 4–6 minutes (installs deps, downloads embedding model on first request)

You'll get a default URL like `https://nepalgov.onrender.com`.

### Step 3 — Verify the default URL works (2 min)

```bash
curl https://nepalgov.onrender.com/api/health
# {"ready": false, ...}   ← still warming
# Wait ~30s after first request
curl https://nepalgov.onrender.com/api/health
# {"ready": true, "stats": {...}}
```

Open the URL in a browser. Should show the editorial landing page.
Submit a query. First request after cold start takes ~30s; subsequent
requests are sub-second.

If this works, you have a deployed demo. The custom domain is optional polish.

### Step 4 — Custom domain via Hostinger (10 min, optional)

#### A. Add the domain in Render

1. Render dashboard → your `nepalgov` service → **Settings** → **Custom Domains**
2. Click **"Add Custom Domain"**
3. Enter: `nepalgov.datacortex.in`
4. Render shows you a CNAME target like `nepalgov.onrender.com`
5. Render also shows a verification token (CNAME or TXT)

#### B. Set the DNS record at Hostinger

1. Log in to Hostinger → **Domains** → `datacortex.in` → **Manage DNS**
2. Click **"Add Record"**

| Type | Name | Points to | TTL |
|------|------|-----------|-----|
| `CNAME` | `nepalgov` | `nepalgov.onrender.com` | 3600 |

3. If Render asked for a verification CNAME/TXT, add that too — exactly as shown.
4. Save.

#### C. Wait for propagation + TLS

DNS propagation usually takes 2–10 minutes on Hostinger. Verify:

```bash
dig +short CNAME nepalgov.datacortex.in
# Should return: nepalgov.onrender.com.
```

Go back to Render's Custom Domains page. It will show:

- ⏳ "Verifying DNS" → wait
- ✅ "Verified, issuing certificate" → wait (Let's Encrypt, ~1 min)
- ✅ "Active" — you're done

Open https://nepalgov.datacortex.in in **incognito**. Should load with
valid TLS and respond to queries.

## Free tier behavior — what to expect

- **Cold start:** ~30 seconds after 15 minutes of no traffic. The frontend
  shows a "Starting the service" message and polls until ready, then retries
  the query automatically.
- **Persistent disk:** Render free tier has no persistent disk. The
  embedding cache rebuilds on every cold start. Tolerable for a demo;
  upgrade to Starter ($7/mo) if you want instant warm starts.
- **Memory:** 512 MB. `multilingual-e5-small` + 5 PDFs + FastAPI overhead
  fits, but with little headroom. If you see OOM in the Render logs,
  upgrade.

## Updates

```bash
# Any push to main with autoDeploy: true triggers a rebuild
git push
# Watch the build in the Render dashboard → Events
```

## Troubleshooting

- **"Application failed to respond"** — backend is still starting (the
  embedding model download happens on first boot). Wait 60s and retry.
- **502 from Render** — check Logs in dashboard. Most common: a Python
  import error from a missing dep in `[demo]` extras.
- **CNAME not verifying** — Hostinger sometimes adds the domain root as a
  suffix. The Name should be just `nepalgov`, not `nepalgov.datacortex.in`.
- **Persistent OOM** — the 512MB free instance is genuinely tight.
  Either upgrade to Starter, or switch to a lighter embedding model
  (`all-MiniLM-L6-v2`, English only — would invalidate the Nepali
  story, not recommended).

## What's intentionally not here

- No CI/CD beyond Render's autodeploy. Single source of truth = main branch.
- No auth — the demo is read-only and rate-limited at Render's edge.
- No analytics — sovereignty-first project, no telemetry.
