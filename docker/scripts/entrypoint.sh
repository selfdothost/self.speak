#!/bin/bash
set -e

if [ "$DOWNLOAD_MODEL" = "true" ]; then
    python download_model.py --output api/src/models/v1_0
fi

# ── Chatterbox WORKER (sibling process) — INTEGRATION-PLAN-v2.md §1.3 ─────────
# Launch the Chatterbox worker in the background under its OWN venv (torch 2.6),
# bound to 127.0.0.1:8881 (localhost-only, never core-facing). It is deliberately
# NOT on the pod liveness path: only the main process's /health is probed, so if
# the worker dies Kokoro serving keeps working. We wrap it in a bounded-backoff
# respawn loop (a minimal supervisor, NOT supervisord) so a crash relaunches it
# without taking the pod down; it reloads its model lazily on the next request.
# Startup is NOT blocked on the ~2 GB HF weight pull. Guarded on the venv existing
# so the CPU/ROCm images (which build no second venv) skip it cleanly.
if [ -x /app/.venv-chatterbox/bin/python ]; then
    (
        set +e
        backoff=1
        while true; do
            echo "[entrypoint] starting chatterbox worker (respawn backoff=${backoff}s)"
            /app/.venv-chatterbox/bin/python -m api.src.chatterbox_worker.main
            echo "[entrypoint] chatterbox worker exited (rc=$?); respawning in ${backoff}s"
            sleep "$backoff"
            if [ "$backoff" -lt 30 ]; then
                backoff=$(( backoff * 2 ))
            fi
        done
    ) &
    echo "[entrypoint] chatterbox worker respawn loop started (pid $!)"
else
    echo "[entrypoint] no /app/.venv-chatterbox — skipping chatterbox worker (Kokoro-only image)"
fi

# ── MAIN process (Kokoro + FastAPI) — the pod's liveness anchor (PID 1-ish) ───
exec uv run --extra $DEVICE --no-sync python -m uvicorn api.src.main:app --host 0.0.0.0 --port 8880 --log-level debug
