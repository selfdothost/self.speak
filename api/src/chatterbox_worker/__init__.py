"""Chatterbox WORKER process package (runs under /app/.venv-chatterbox, torch 2.6).

This package is imported ONLY by the sibling Chatterbox worker process, never by
the main (Kokoro, torch 2.8) uvicorn process. It must therefore import nothing
from the main app's inference stack (kokoro / KokoroV1 / ModelManager) — its only
heavy dependency is ``chatterbox-tts`` (+ its torch 2.6), which lives exclusively
in the chatterbox venv. See INTEGRATION-PLAN-v2.md §1.2 / §1.4 for the two-process
topology and the localhost control API this package serves on 127.0.0.1:8881.

Deliberately empty of eager imports so ``python -m api.src.chatterbox_worker.main``
does not drag torch in until ``main`` asks for it.
"""
