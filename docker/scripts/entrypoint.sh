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
            started=$(date +%s)
            /app/.venv-chatterbox/bin/python -m api.src.chatterbox_worker.main
            rc=$?
            ran=$(( $(date +%s) - started ))
            echo "[entrypoint] chatterbox worker exited (rc=${rc}) after ${ran}s; respawning in ${backoff}s"
            sleep "$backoff"
            # RESET the backoff after a worker that actually ran. Without this it
            # only ever grows: six exits pin it at 30s for the life of the pod,
            # and every later restart -- including a healthy one -- waits half a
            # minute. That matters now that a FORCED VRAM release deliberately
            # exits this worker to return its CUDA context (see
            # api/src/chatterbox_worker/reclaim.py): those are cooperative,
            # expected exits after minutes of healthy service, and must not be
            # counted as crash-looping. The backoff exists for a worker that dies
            # immediately and repeatedly; 60s of uptime is the line between the
            # two.
            if [ "$ran" -ge 60 ]; then
                backoff=1
            elif [ "$backoff" -lt 30 ]; then
                backoff=$(( backoff * 2 ))
            fi
        done
    ) &
    echo "[entrypoint] chatterbox worker respawn loop started (pid $!)"
else
    echo "[entrypoint] no /app/.venv-chatterbox — skipping chatterbox worker (Kokoro-only image)"
fi

# ── Kokoro WORKER (sibling process) — self.speak#5 P2, DEFAULT OFF ───────────
# Same respawn-loop shape as the chatterbox worker above, but gated on
# KOKORO_WORKER_ENABLED because turning it on before the P3 cutover makes VRAM
# strictly WORSE, not better: main still imports Kokoro and holds its own CUDA
# primary context, so running the worker too means TWO contexts (~470 MiB each)
# where there was one. The saving only arrives when main stops importing Kokoro
# and this becomes the sole holder.
#
# Runs from the MAIN venv on purpose. The chatterbox worker needs
# /app/.venv-chatterbox only because of a torch-version conflict; Kokoro has no
# such conflict, so a separate PROCESS is all that is needed and a whole class
# of build complexity disappears.
if [ "$KOKORO_WORKER_ENABLED" = "true" ]; then
    (
        set +e
        backoff=1
        while true; do
            echo "[entrypoint] starting kokoro worker (respawn backoff=${backoff}s)"
            started=$(date +%s)
            uv run --extra $DEVICE --no-sync python -m api.src.kokoro_worker.main
            rc=$?
            ran=$(( $(date +%s) - started ))
            echo "[entrypoint] kokoro worker exited (rc=${rc}) after ${ran}s; respawning in ${backoff}s"
            sleep "$backoff"
            # Reset after a worker that actually ran -- see the chatterbox loop
            # above for why this is not optional. A forced VRAM release exits
            # this worker deliberately, and those cooperative exits must not be
            # counted as crash-looping or availability degrades to 30s.
            if [ "$ran" -ge 60 ]; then
                backoff=1
            elif [ "$backoff" -lt 30 ]; then
                backoff=$(( backoff * 2 ))
            fi
        done
    ) &
    echo "[entrypoint] kokoro worker respawn loop started (pid $!)"
else
    echo "[entrypoint] KOKORO_WORKER_ENABLED != true — kokoro stays in-process (pre-P3 default)"
fi

# ── MAIN process (Kokoro + FastAPI) — the pod's liveness anchor (PID 1-ish) ───
exec uv run --extra $DEVICE --no-sync python -m uvicorn api.src.main:app --host 0.0.0.0 --port 8880 --log-level debug
