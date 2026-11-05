# timbrica-reuse-worker

RunPod serverless worker for /voice-enhancer «Максимальная очистка»: NVIDIA RE-USE
(`nvidia/NVIDIA-NemotronLabs-RE-USE`, revision `31f82f2e…`, 9.6 M parameters, OpenMDW-1.1 — commercial use and
derivatives allowed; the LICENSE ships inside the image at `/models/reuse/LICENSE`) — noise, room echo, clipping and
band limitation removed from speech, 8–48 kHz.

Lives under `services/reuse-worker/` in the app repo and is mirrored to `github.com/bandidas1/timbrica-reuse-worker`,
whose CI (`.github/workflows/build.yml`) runs the transfer self-test and pushes
`ghcr.io/bandidas1/timbrica-reuse-worker:<commit>`. Pin the endpoint to the COMMIT TAG (a re-pushed `:latest` does not
roll an idle worker — see services/clip-warp-worker/README.md). Nothing from the Laravel app is imported here.

## Contract (keep in lockstep with the app's provider)

- input `{"source_url", "result_url", "deadline_at"}`: one signed GET (WAV/FLAC, 8–48 kHz, 1–2 ch), one signed PUT of
  the result (FLAC 24-bit, same rate and channels). The app's PUT sink is idempotent, so `reuse_io.upload` retries a
  dropped connection / 408 / 425 / 429 / 5xx until the deadline; 4xx refusals are final.
- input `{"probe": true}` → `{"ok": true, "probe": true, "gpu": …}` at once (`RunPodApi::probeServes`).
- errors: `missing_urls`, `source_unreadable`, `bad_rate`, `too_long`, `deadline_passed`, `gpu_oom`,
  `source_*` / `result_*` (transfer).

## Measured (05.11.2026, app repo `docs/voice-enhancer-meter.md`)

- RTX 4090, official mamba_ssm kernels: 48 kHz — 6.4 GPU-s per audio minute with 20-s chunks / 10 % overlap (same
  quality as NVIDIA's 5 s / 50 %: 8.5 GPU-s), 16 kHz — 2.45. Peak VRAM 11 GB at 48 kHz → 24 GB pools only.
- CPU is not an option: 7.8–13.4 s per audio second on 16 threads even with a compiled scan.
- Quality vs the in-browser chain (DeepFilterNet 3): DNSMOS 3.47 vs 3.32 on iPhone-hiss speech, 3.29 vs 2.98 on the
  degraded corpus, voice similarity 0.867 vs 0.812; highs 4–20 kHz on the voice within ±1.4 dB of a clean VCTK
  reference.
- Image base: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (cu124 → no sm_120: keep Blackwell MIG
  out of the endpoint's pools, as clip-warp does).
