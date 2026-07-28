---
created: "2026-07-24"
last_edited: "2026-07-24"
---

# Implementation Tracking: vram-lease-client (R1 VRAM State / R2 Release Handling)

Build site: context/plans/build-site-vram-lease-client.md
Cavekit: context/kits/cavekit-vram-lease-client.md (R1 VRAM State Reporting,
R2 Release-Request Handling). self.speak's side of the cross-service VRAM-lease
protocol brokered by core (`selfai/self.ai`).

All 9 tasks landed. Everything is in the Python `api/src/` layer of the single
uvicorn process — no C++ router, no supervisord split (the two divergences from
the self.llamolotl sibling). All GPU introspection is `torch.cuda`; all
single-flight / drain-wait synchronisation is in-process on the asyncio loop.

Validation in this environment: `python -m py_compile` clean on every touched
file; `ruff check .` (import-sort `I` only) — **All checks passed**. Full
`pytest` needs the heavy deps (torch/kokoro/fastapi) → CI (`test:pytest`,
`.[test,cpu]`) is the real gate. The probe tri-state, drain-wait, and the entire
release orchestration were nonetheless exercised **live** here against the real
`vram_lease.py` (torch stubbed for the mocked GPU cases) — all logic checks
passed. The wire models were round-tripped live under pydantic 2.13 — all passed.
The endpoint/auth tests (T-007/T-008) require the full app import and run in CI.

| Task | Status | Notes |
|------|--------|-------|
| T-001 | BUILT | `api/src/core/auth.py`: extended the docstring scope taxonomy to add `system:read - GET /api/system/vram-state` and `system:write - POST /api/system/vram-release` (reusing llamolotl's exact names), with an explicit note that core's minting side must ADD audience `self.speak` to the same two scopes and the strings must match byte-for-byte. Added the optional named constants `SCOPE_AUDIO_SYNTHESIZE` / `SCOPE_SYSTEM_READ` / `SCOPE_SYSTEM_WRITE` so T-007/T-008 route decorators reference constants, not bare strings. No enum/allow-list exists (`require_scope` is a plain membership check), so "adding a scope" is exactly this docstring+constants change — verified against auth.py:106-134. Tested in T-009 (`TestVramStateEndpoint`/`TestVramReleaseEndpoint` wrong-scope→403, correct-scope not-rejected). |
| T-002 | BUILT | `api/src/inference/vram_lease.py` (new) — `probe_vram_state() -> dict`, computed live every call (no caching). Tri-state exactly per R1-AC4: (a) `is_available()` True + `mem_get_info()`/`memory_allocated()` succeed → real ints (`status="ok"`, near-zero held not null when idle); (b) `is_available()` True but a CUDA call **raises** → `held`/`capacity` `None`, `status="unreachable"`, never 0 (R1-AC5); (c) `is_available()` False → integer `0`, `status="no_gpu"`. The `try/except` around the CUDA calls is load-bearing (self.transcribe first-call-throw history) and even wraps `is_available()` itself. Also returns `model_resident` (reads `ModelManager._instance._backend.is_loaded` best-effort, never constructs a manager, never fatal). Live-verified: cases a/b/c and the liveness (changing `memory_allocated` → changed value, uncached). Case (c) runs unmocked in CI. |
| T-003 | BUILT | `api/src/structures/schemas.py`: `VramStateResponse(BaseModel)` — `held_vram_bytes: Optional[int]`, `total_capacity_bytes: Optional[int]` (Optional so `None` serialises to JSON `null`, not a forced false-0), `gpu_reachable: bool`, `status: str`, `model_resident: bool`. Every field's unit (bytes) and null-vs-int meaning documented in `Field(description=...)` precisely enough for core's transport to parse with no translation (R1-AC6). Live-verified: `held=None` → JSON `null` (not 0); int fields stay JSON ints; field names exact. |
| T-004 | BUILT | `api/src/structures/schemas.py`: `VramReleaseRequest` (`target_bytes: int`, `timeout_seconds: float`, **no mechanism field** — matches core's outbound body verbatim) and `VramReleaseResponse` (`status: Literal["released","partial","busy"]`, `freed_bytes: int`). `freed_bytes` typed `int` so a JSON boolean never reaches the wire (core's `_map_response` does `isinstance(freed,int) and not isinstance(freed,bool)`). Live-verified under pydantic 2.13: request round-trips; `freed_bytes=True` **coerces to int `1`** (repo's pinned 2.10 may instead reject — either way a boolean never survives to the wire, so the AC holds); `status="denied"` → ValidationError. |
| T-005 | BUILT | `api/src/inference/vram_lease.py`: module-level `_synthesis_count` (plain int, atomic on the single loop thread), `synthesis_in_flight()`, async CM `track_synthesis()` (increments on enter, decrements in `finally` — survives a raise AND a GeneratorExit), and `async wait_for_drain(timeout_seconds) -> bool`. **Deviation (documented):** drain uses a bounded `asyncio.sleep` **poll loop** (`_DRAIN_POLL_INTERVAL=0.02s`), explicitly one of the two options the build site offered, chosen over a module-level `asyncio.Event` because a persistent Event would bind to one test's event loop and raise "bound to a different loop" under `--asyncio-mode=auto` across tests. `api/src/services/tts_service.py`: `generate_audio_stream` now wraps its whole lifetime in `async with track_synthesis()`. **Deviation (documented):** rather than re-indent the ~130-line generator body, the original body was renamed to `_generate_audio_stream_impl` and `generate_audio_stream` is a thin wrapper that `async with track_synthesis(): async for chunk in self._generate_audio_stream_impl(...): yield chunk`. The CM spans the full generator lifetime (exhaustion or close), so the counter never leaks; `generate_audio` still flows through the wrapper (counted once, no double-count). Live-verified: idle→True fast; held→False past a 50ms timeout; released-early→True; counter decremented on a raising body. |
| T-006 | BUILT | `api/src/inference/vram_lease.py`: `async def handle_vram_release(target_bytes, timeout_seconds) -> VramReleaseResponse` + `_unload_backend()`. Sequence: (1) **AC7 single-flight** — module-level bool `_release_in_flight`, check-and-set with no await between (**deviation:** a bool flag instead of a non-blocking `asyncio.Lock` acquire — functionally identical on the single loop thread, avoids an asyncio.Lock binding to a stale test loop; noted). (2) **AC8** `no_gpu` → `released`/0 honest no-op; `unreachable` (case b) → `partial`/0. (3) **AC3** live baseline via `probe_vram_state()`. (4) **AC2/AC6** `await wait_for_drain(timeout)`; still-generating at deadline → `partial`/0 with **no unload** (never yank the model). (5) free via `_unload_backend()` which reads `ModelManager._instance` directly, unloads only if resident, never constructs a manager, swallows an unload raise (freed comes from the live delta, not the call). (6) **AC3/AC5** re-probe; `freed = max(0, before-after)` from the live delta; `released` iff `freed >= target` else truthful `partial`. Live-verified: AC3 unload-frees-nothing→0/partial; AC2/AC6 no-unload/partial within budget; AC5 smaller→real partial; happy→released; AC7 busy no-work; AC8 cpu→released/0 no unload; case-b→partial/0 no unload. |
| T-007 | BUILT | `api/src/routers/system.py` (new, `APIRouter(tags=["system"])`): `GET /api/system/vram-state`, `Depends(require_scope(SCOPE_SYSTEM_READ))`, `async`. Calls `probe_vram_state()`, marshals into `VramStateResponse(**probe)` (probe dict keys are exactly the model fields). **Always 200** (R1-AC3): case b/c carry the signal in the body; an unexpected probe raise is caught → 200 `unreachable` with null figures, never a 500. Registered in `api/src/main.py` (`from .routers.system import router as system_router`; `app.include_router(system_router)` next to `control_router`; no prefix — full path on the decorator). Tested in T-009 (`TestVramStateEndpoint`: 200 across a/b/c, exception→200 unreachable, no-ticket→401, wrong-scope→403). |
| T-008 | BUILT | `api/src/routers/system.py`: `POST /api/system/vram-release`, `VramReleaseRequest` body, `Depends(require_scope(SCOPE_SYSTEM_WRITE))`, `async` (awaits the drain). `await handle_vram_release(...)`; `status=="busy"` → `HTTPException(409)` (the 409 core's `_map_response` reads as "not now"); else returns the `VramReleaseResponse` (200). Tested in T-009 (`TestVramReleaseEndpoint`: valid→200 freed body, busy→409, no-ticket→401, wrong-scope→403, missing `target_bytes`→422). |
| T-009 | BUILT | `api/tests/test_vram_lease.py` (new). Mirrors `test_ticket_auth.py` conventions: `TestClient(app)` never entered as a `with` (no lifespan/model load), `mint_test_ticket(scope=...)` for auth, `SERVICE_AUTH_SECRET` from `conftest.py`. Covers all 14 ACs: probe a/b/c + liveness (case c unmocked-live in CI); wire models null→null/int/bool-never-on-wire/status-vocab; drain idle/held/released/raise; release AC3/AC2/AC6/AC5/happy/AC7/AC8/case-b + an unmocked CPU release; endpoints 200-across-tri-state, 401/403 auth, 409-on-busy, 422-on-missing-field. GPU cases patch `vram_lease.torch.cuda`; the probe/release module-fn seams patch `api.src.inference.vram_lease.*` and the endpoints patch `api.src.routers.system.*`. |

## Deviations from the build site (all deliberate, none change a wire contract)

1. **T-005 drain = asyncio.sleep poll loop, not an asyncio.Event.** The build
   site offered either; the poll loop avoids a module-level Event binding to a
   stale event loop across `--asyncio-mode=auto` tests. Bounded by the
   wall-clock deadline, never blocks the loop.
2. **T-005 tts_service wrap = wrapper-delegates-to-renamed-impl, not an
   in-place body re-indent.** Same CM semantics (spans exhaustion + close), far
   smaller/safer diff. `generate_audio` still routes through the wrapper.
3. **T-006 single-flight = module bool, not asyncio.Lock.** Check-and-set is
   await-free on the single loop thread, so it is race-free and functionally a
   non-blocking acquire, without an asyncio.Lock loop-binding hazard in tests.

## Cross-repo handoffs (NOT built here — must land in `selfai/self.ai`)

1. Mint audience `self.speak` on the `system:read`/`system:write` scopes
   (`api/selfai_ui/utils/service_auth.py`) — scope strings must match T-001
   byte-for-byte. Core already mints this pair for `self.llamolotl`.
2. `vram_speak.py` release transport (the `vram_llamolotl.py` analogue) —
   mints a `system:write` ticket for `self.speak`, POSTs
   `{target_bytes, timeout_seconds}`, maps the reply defensively (409/busy →
   TIMEOUT; recognised status + int `freed_bytes` → CONFIRMED).
3. Config-driven registration (`SPEAK_VRAM_CAPACITY_BYTES` +
   `_register_speak_vram_consumer()`).
4. The self.speak GPU deployment manifest + `SPEAK_VRAM_CAPACITY_BYTES` wiring.

## Caveats / not verified here

- **Full pytest not run in this environment** (no torch/kokoro/fastapi). CI
  (`test:pytest`, `allow_failure: false`, `.[test,cpu]`) is the gate. The probe
  tri-state, drain-wait, and full release orchestration WERE exercised live
  against the real `vram_lease.py` (torch stubbed for the GPU-mocked cases); the
  wire models were round-tripped live. The FastAPI endpoint/auth tests
  (T-007/T-008) import the whole app and run only in CI.
- **`freed_bytes=True` coercion is pydantic-version-dependent** (2.13 here
  coerces to `1`; the repo pins 2.10 which may reject). The AC — "a boolean
  never reaches the wire" — holds in both cases; the T-009 test accepts either.
- **Single-flight is per-process, not cross-replica.** Sufficient for this
  single-replica phase; a multi-replica serving plane would need a shared lock
  (out of scope, noted so it isn't silently assumed away).
