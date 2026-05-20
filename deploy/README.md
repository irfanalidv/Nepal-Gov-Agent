# Deploy: nepalgov.datacortex.in

Runbook for the hosted demo on your existing Stacksift VPS. Assumes Ubuntu
22.04+/Debian 12+, Caddy already installed and running (since Stacksift is
already on this box).

## One-time setup

```bash
# 1. DNS — point nepalgov.datacortex.in (A record) at the VPS public IP.
#    Wait for propagation (usually <5 min on Cloudflare).

# 2. Code + venv
sudo mkdir -p /srv/nepalgov
sudo chown -R $USER:$USER /srv/nepalgov
cd /srv/nepalgov

git clone https://github.com/irfanalidv/Nepal-Gov-Agent.git .
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[demo]"

# 3. Corpus — pull seed PDFs once.
python -c "from nepal_gov_agent import download_corpus; download_corpus('./Data')"

# 4. Pre-warm the embedding cache so first request isn't cold (~30s build).
python -c "
from nepal_gov_agent import GovRAG, GovRAGConfig
rag = GovRAG(corpus_dir='./Data', config=GovRAGConfig(cache_dir='.cache'))
print(rag.stats)
print(rag.ask('test query').confidence)
"

# 5. Permissions for systemd user
sudo chown -R www-data:www-data /srv/nepalgov
sudo mkdir -p /srv/nepalgov/sessions
sudo chown www-data:www-data /srv/nepalgov/sessions

# 6. systemd
sudo cp deploy/nepalgov.service /etc/systemd/system/nepalgov.service
sudo systemctl daemon-reload
sudo systemctl enable --now nepalgov.service
sudo systemctl status nepalgov.service
# Tail logs: journalctl -u nepalgov -f

# 7. Caddy
# If you use a single Caddyfile, append the block from deploy/Caddyfile.nepalgov.
# If you use sites-enabled style:
sudo cp deploy/Caddyfile.nepalgov /etc/caddy/sites-available/nepalgov.datacortex.in
sudo ln -sf /etc/caddy/sites-available/nepalgov.datacortex.in /etc/caddy/sites-enabled/
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy

# 8. Smoke test
curl -I https://nepalgov.datacortex.in
# 200 OK with valid TLS = done.
```

## Updates

```bash
cd /srv/nepalgov
sudo -u www-data git pull
sudo -u www-data .venv/bin/pip install -e ".[demo]"
sudo systemctl restart nepalgov
```

## Troubleshooting

- **502 Bad Gateway** → `journalctl -u nepalgov -n 100` — most likely the
  app is still loading the embedding model (~30s first boot) or
  sentence-transformers is downloading. Wait, then retry.
- **Embeddings re-downloading on every restart** → set `HF_HOME` (already
  set in the systemd unit). Confirm with `ls /srv/nepalgov/.cache/huggingface`.
- **WebSocket disconnects** → Caddy handles WS by default; check
  `journalctl -u caddy` for upstream errors.
- **High memory** → the model + embeddings sit ~700MB RSS. If you see >2GB
  consistently, drop `MemoryMax` in the unit file to force OOMs you can
  see, then investigate (likely a leak in a dependency).

## What's intentionally not here

- No auth on the demo. It's read-only and rate-limited at the app level
  by Gradio's queue. If abuse becomes a problem, add Caddy basic auth or
  a Cloudflare rule.
- No autoscaling. One process, one VPS. Fine for the traffic this will see.
- No CI/CD. Pull + restart is enough for a maintenance-mode project.
