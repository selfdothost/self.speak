---
created: "2026-07-24"
last_edited: "2026-07-24"
---

# Cavekit: VRAM Lease Client

## Scope
self.speak's side of a cross-service VRAM-lease protocol whose broker lives in
core (selfai-api, the `self.ai` repo) — see `selfai/self.ai`'s
`cavekit-gpu-lease-broker.md`, whose consumer-integration requirement depends on
this kit's interface. It is the audio-backend counterpart to self.llamolotl's
already-built `cavekit-gpu-lease-client.md` (VRAM state reporting + release
handling), mirroring the same R1/R2 decomposition and the same two endpoints
(`GET /api/system/vram-state`, `POST /api/system/vram-release`), adapted to
self.speak's realities. This kit adds two capabilities: (1) reporting the VRAM
this process currently holds plus its total addressable capacity so core's lease
registry can track this instance, and (2) accepting a release-request (free N
bytes) and answering it honestly.

**Where this DIVERGES from the llamolotl sibling — named, not hand-waved:**

1. **No `/api/system/*` surface exists yet.** self.speak's only control
   endpoint today is `GET /api/voices` (`api/src/routers/control.py`, unticketed
   voice-catalog alias). This kit introduces the `/api/system/*` namespace to
   this repo for the first time.
2. **Model residency is fundamentally different.** self.llamolotl hosts N whole
   models and releases by evicting whole models in LRU order. self.speak holds
   exactly ONE Kokoro backend as a module singleton
   (`api/src/inference/model_manager.py` — `ModelManager._instance`, wrapping a
   single `KokoroV1`, ~330MB, loaded at warmup and always resident). There is no
   LRU set to evict from. What self.speak *frees* on request is therefore a
   genuine design decision, not a mechanical LRU loop — enumerated in
   **Open Questions**, kept agnostic here. Whatever it frees, R2's ACs require
   the reported figure be **honest** (measured live via `torch.cuda`
   before/after, never assumed) and a release that cannot hit the target say so
   truthfully rather than fabricate a success.
3. **Probe mechanism has no router to cross-check.** self.llamolotl cross-checks
   its C++ router's HTTP API against `nvidia-smi`. self.speak has no such router;
   it probes its own process's held VRAM directly (`torch.cuda.memory_allocated`
   / `mem_get_info`). Critically, self.speak runs BOTH a CPU deployment (today)
   and a forthcoming GPU deployment, so the probe must handle the no-CUDA case
   as a first-class, honest answer (see R1-AC4's tri-state), not an error.
4. **Simpler process model.** A single uvicorn process with module-singleton
   state — no supervisord+router split. Single-flight (R2-AC6) is an in-process
   lock (`threading.Lock` / asyncio lock), simpler than llamolotl's.

The request core sends is **amount-based only** (a target byte count, no
mechanism field) — the same design property the llamolotl kit calls out — so
self.speak decides HOW to free, and a future multi-engine self.speak
(Chatterbox / GPT-SoVITS resident alongside Kokoro) is a same-protocol upgrade,
not a wire change.

## Requirements

### R1: VRAM State Reporting
**Description:** Exposes the VRAM this self.speak process currently holds and its
total addressable GPU capacity through an authenticated endpoint core's lease
registry can query. Computes both figures with a live GPU query at query time
(`torch.cuda.memory_allocated` / `mem_get_info`), never a cached value. Handles
the no-CUDA (CPU) deployment as a first-class honest answer — a CPU self.speak
registers as a zero-capacity, no-op consumer rather than erroring.
**Acceptance Criteria:**
- [ ] An authenticated endpoint `GET /api/system/vram-state` reports currently-held
      VRAM in bytes and total addressable capacity in bytes. It is gated by the
      ticket scope **`system:read`** (decided — see Decisions; matches llamolotl's
      taxonomy exactly). self.speak's scope taxonomy (`api/src/core/auth.py`) today
      holds ONLY `audio:synthesize`, so `system:read` MUST be ADDED to that
      taxonomy — a NEW scope for self.speak, unlike llamolotl which already had it
      (see the cross-repo dependency below).
- [ ] The held and capacity figures are computed live per request via a direct
      GPU query, never served from a cached or stored value — a stale answer
      defeats the broker's purpose.
- [ ] The endpoint returns HTTP 200 on every call, including when the GPU is
      unreachable or absent — the reachability signal lives in the response body,
      never in an HTTP error that would hide it.
- [ ] The response distinguishes three states without ambiguity: (a) **GPU
      reachable and probed** → `held_vram_bytes` and `total_capacity_bytes` are
      real integers (held near-zero, not null, when the model is not resident);
      (b) **GPU present but unprobable/unreachable** → both fields are **null,
      not 0**, with a status marking it unreachable (so the caller cannot mistake
      "unknown" for "genuinely near-zero"); (c) **no CUDA / CPU deployment** →
      both fields are the integer **0** with a status marking it non-GPU (a
      known-zero, no-op consumer — distinct from the null/unreachable case).
- [ ] An unreachable or unknown held figure is NEVER coerced to 0 — a false zero
      would let core over-grant VRAM that is not actually free.
- [ ] The wire format is documented precisely enough that core's vram-state
      transport parses held/capacity directly with no translation: units are
      **bytes**, field names are `held_vram_bytes` and `total_capacity_bytes`,
      and their null-vs-integer semantics per R1-AC4 are stated explicitly.
**Dependencies:** Cross-repo (`selfai/self.ai`, `cavekit-gpu-lease-broker.md`) —
core must (a) mint a read scope for audience `self.speak` on its ticket-minting
side (`api/selfai_ui/utils/service_auth.py`); the scope NAME must match what R1
adds to `auth.py`, an explicit two-sided coordination, and (b) register
self.speak as a lease consumer (config-driven, e.g. `SPEAK_VRAM_CAPACITY_BYTES`
+ a `_register_speak_vram_consumer()`). Both live in the self.ai repo and are NOT
built by this kit.

### R2: Release-Request Handling
**Description:** Accepts an authenticated release-request specifying a target
amount of VRAM to free and a timeout, and answers it honestly. **Mechanism
(decided — see Decisions): wait-then-free within the timeout.** On a release
request self.speak waits — bounded by `timeout_seconds` — for any in-flight
synthesis to drain, then unloads the Kokoro backend and frees its VRAM,
reporting the live-measured freed amount as a confirmed release. If the timeout
is reached while a synthesis is still generating (the model cannot be safely
pulled out from under an active request), it returns a truthful `partial` with
the real freed figure (typically 0) rather than yanking the model or fabricating
a success. Regardless, the *reported* freed amount must reflect a live GPU
measurement, the reply must use the exact shape core recognizes as a confirmed
release, and anything less than an unambiguous confirmed release must be honest
about the uncertainty. The unloaded backend must lazily cold-reload on the next
synthesis (R2-AC8) — an accepted latency cost of yielding VRAM, NOT a permanent
break.
**Acceptance Criteria:**
- [ ] An authenticated endpoint `POST /api/system/vram-release` accepts a body of
      `{target_bytes: int, timeout_seconds: float}` — amount-based only, with NO
      mechanism field (self.speak decides how to free), matching the shape core's
      release-request protocol sends. It is gated by the ticket scope
      **`system:write`** (decided; matches llamolotl). As with R1's `system:read`,
      `system:write` MUST be ADDED to `api/src/core/auth.py`'s taxonomy — a NEW
      scope for self.speak.
- [ ] When a synthesis is in flight at request time, the handler **waits for it to
      drain, bounded by `timeout_seconds`**, before unloading — it never drops the
      resident model out from under an active `POST /v1/audio/speech`. If the
      synthesis finishes within the window, the backend is unloaded and the freed
      VRAM is confirmed; if the deadline is hit first, the reply is a truthful
      `partial` with the real (typically zero) freed figure. When self.speak is
      idle at request time, it frees immediately without waiting.
- [ ] The `freed_bytes` reported is measured via a live GPU query (`torch.cuda`)
      taken before and after the free attempt — never assumed from the fact that
      an unload/free call returned successfully.
- [ ] On the happy path the reply is the recognizable confirmed shape:
      `{status: "released", freed_bytes: <int>}` when the target was met, or
      `{status: "partial", freed_bytes: <int>}` when a real but smaller amount was
      freed. `status` is one of the values core recognizes as confirmed
      (`released` / `partial` / `confirmed`) and `freed_bytes` is a JSON integer
      — never a boolean (core rejects a bool as a mistyped `freed_bytes`).
- [ ] A release that cannot free the target is truthful — a `partial` status with
      the real, smaller freed figure, or a truthful small/zero freed figure — and
      NEVER a fabricated success that claims the target was freed when it was not.
- [ ] The endpoint responds within the requested `timeout_seconds` with the
      actually-confirmed freed amount (which may be less than target, or zero) —
      it never blocks past the caller's timeout waiting on a slow or stuck free.
- [ ] At most one release-request is processed at a time, enforced by an
      in-process lock (single uvicorn process). A second concurrent request is
      answered with **HTTP 409** (or an in-band `{status: "busy"}`), never raced
      against the first — core reads both as "not confirmed / try later", never a
      denial or a success.
- [ ] On a CPU / no-CUDA deployment (nothing to free) a release-request is
      answered honestly with a truthful zero freed and a recognizable status — an
      empty-but-valid no-op — never an error and never a fabricated success.
- [ ] **After a release unloads the backend, the NEXT synthesis transparently
      cold-reloads it — synthesis must NOT stay broken.** The model is loaded once
      at startup warmup; a release drops it, so the serving path must lazily
      re-initialize when the backend is absent instead of failing "Backend not
      initialized" permanently until pod restart. The reload is serialized (a
      burst after a release triggers exactly one reload) and skips the startup
      warmup synthesis (the real request is the warmup). A test must exercise
      serve-AFTER-release end to end — not a mocked unload — since that is the
      exact gap that let this ship broken (found only in live GPU verification).
technique). Cross-repo (`selfai/self.ai`) — core must (a) mint a write scope for
audience `self.speak` (`service_auth.py`), scope name coordinated with R2's
`auth.py` addition, and (b) implement the outbound release transport for the
`self.speak` consumer (the self.speak analogue of `vram_llamolotl.py`, e.g.
`vram_speak.py`), which POSTs to this endpoint and maps the reply defensively.
Both live in the self.ai repo and are NOT built by this kit.

## Out of Scope
- **Chatterbox / GPT-SoVITS multi-engine residency** — the forthcoming voice
  engines (`cavekit`-tracked elsewhere) are not in scope. R2's amount-only,
  mechanism-free request is deliberately designed so a future multi-engine
  self.speak frees differently with no protocol change; that build is separate.
- **Training / DeepSpeed contention** — self.speak is inference-only; there is no
  training-vs-inference contention guard dependency here (unlike a future
  llamolotl phase). This kit's consumer participates in the lease protocol for
  serving VRAM only.
- **Self-registration with core's lease registry** — core registers self.speak
  config-driven (`SPEAK_VRAM_CAPACITY_BYTES`) for this single-instance phase;
  this kit does not build dynamic self-registration.
- **The core-side broker, transport, scope-minting, and consumer registration** —
  registry, release-request issuing, grant protocol, `service_auth.py` scope
  minting for audience `self.speak`, `_register_speak_vram_consumer()`, and the
  `vram_speak.py` release transport are all `selfai/self.ai`'s
  `cavekit-gpu-lease-broker.md`, a separate coordinated repo/kit this one is the
  counterpart to. This kit's two endpoints are the authoritative contract that
  work implements against.
- **The self.speak GPU deployment manifest** and the actual
  `SPEAK_VRAM_CAPACITY_BYTES` wiring — separate deploy work; this kit specifies
  only the in-process endpoint behavior, which must be correct on both the CPU
  (today) and GPU (forthcoming) deployments.

## Decisions
Both open questions were resolved by the maintainer on 2026-07-24, before this
kit went to the architect:

- **Release mechanism (R2): wait-then-free within the timeout.** On a release
  request, wait — bounded by `timeout_seconds` — for any in-flight synthesis to
  drain, then unload the Kokoro backend (`ModelManager.unload_all()` /
  `KokoroV1.unload()`, which drops the model, clears pipelines, and calls
  `torch.cuda.empty_cache()`) and confirm the live-measured freed amount. If the
  deadline hits while still generating, return a truthful `partial` (typically
  `freed_bytes: 0`) — never yank the model from under an active request, never
  fabricate a success. Chosen over "unload-if-idle-else-busy" (gives back VRAM
  less reliably under load) and "never unload / report-only" (yields nothing);
  the accepted cost is a cold reload + warmup on the next synthesis after a
  release. This generalizes cleanly to the future multi-engine self.speak, where
  "drain then free" applies per-engine.
- **Scope names (R1/R2): `system:read` / `system:write`.** Reuse llamolotl's
  exact names for cross-backend consistency — core already mints this pair for
  the llamolotl audience, so one taxonomy covers both control planes and the
  minting side needs only to add audience `self.speak` to the same scopes.
  Still a NEW scope *for self.speak* (its taxonomy had only `audio:synthesize`),
  so both `auth.py` and core's `service_auth.py` must gain it for this audience.

## Cross-References
- See also: `selfai/self.llamolotl`'s `cavekit-gpu-lease-client.md` — the
  already-built sibling this kit mirrors (same R1/R2, same two endpoints); the
  divergences (no `/api/system/*` surface, single-singleton residency, no-CUDA
  CPU case, in-process single-flight) are enumerated in Scope.
- See also: `selfai/self.ai`'s `cavekit-gpu-lease-broker.md` — core's side of
  both calls: the vram-state transport that parses R1's body, the release
  transport that maps R2's reply (confirmed vs. busy/timeout), the `service_auth`
  scope minting for audience `self.speak` (R1/R2's cross-repo scope dependency),
  and the config-driven consumer registration.
- Repo context: `api/src/core/auth.py` (ticket auth, scope taxonomy to extend),
  `api/src/routers/control.py` (existing `/api/*` control surface),
  `api/src/inference/model_manager.py` + `kokoro_v1.py` (the Kokoro singleton and
  its existing `unload()` / `torch.cuda` memory calls R1/R2 build on).

## Changelog
- 2026-07-24: Kit created for the self.speak side of the cross-service VRAM-lease
  protocol — VRAM state reporting (R1) and honest, single-flight release-request
  handling (R2), the audio-backend counterpart to self.llamolotl's
  `cavekit-gpu-lease-client.md` and the consumer half of `selfai/self.ai`'s
  `cavekit-gpu-lease-broker.md`.
- 2026-07-24: Resolved both open questions before architecting — release
  mechanism = wait-then-free within timeout; scope names = `system:read` /
  `system:write` (reused from llamolotl). Open Questions section replaced with
  Decisions; R1-AC1, R2 Description, and R2-AC1 updated to lock the choices, and
  a new R2 AC added for the synthesis-drain wait.
- 2026-07-25: REVISE (bug found in live GPU verification, self.ai!204). R2 said
  "the unloaded backend lazily cold-reloads on the next synthesis" as a
  parenthetical FACT, but it was never a requirement and the code never did it —
  so after a release, every synthesis 500'd "Backend not initialized" forever
  until pod restart (only the GPU pod, which actually unloads). Promoted it to
  R2-AC8: the serving path MUST lazily re-initialize when the backend is absent,
  serialized, warmup-skipped, with a serve-AFTER-release test (not a mocked
  unload — the exact gap that let it ship). Fix: ModelManager.ensure_loaded()
  called at the top of the synthesis funnel.
