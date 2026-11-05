# RunPod serverless handler: NVIDIA RE-USE speech enhancement for /voice-enhancer «Максимальная очистка».
#
# Input  {"source_url": signed GET (WAV/FLAC, <= 48 kHz, 1-2 ch), "result_url": signed PUT,
#         "deadline_at": unix seconds the app stops waiting (optional)}
#        {"probe": true} -> answers at once (App\Services\RunPod\RunPodApi::probeServes).
# Output {"ok": true, "sr", "channels", "processed_channels", "audio_s", "gpu_s", "bytes", "gpu"} or {"error": code}.
#
# The result is the cleaned voice at the source's own rate and channel count, FLAC 24-bit; loudness, limiter and
# output format stay in the page's polish chain (the same one DeepFilterNet's output goes through).
# Measured 05.11.2026 (docs/voice-enhancer-meter.md): 6.4 GPU-s per audio minute at 48 kHz on an RTX 4090 with
# 20-s chunks / 10 % overlap — the same quality as NVIDIA's 5-s / 50 % recipe, 25 % faster.
import math
import os
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

import reuse_io

MODEL_DIR = os.environ.get('MODEL_DIR', '/models/reuse')
sys.path.insert(0, MODEL_DIR)
from models.stfts import mag_phase_stft, mag_phase_istft          # noqa: E402  (NVIDIA's code, baked in)
from models.generator_SEMamba_time_d4 import SEMamba               # noqa: E402
from utils.util import pad_or_trim_to_match                        # noqa: E402

CHUNK_S = float(os.environ.get('REUSE_CHUNK_S', '20'))
HOP_P = float(os.environ.get('REUSE_HOP_P', '0.9'))
MIN_SR, MAX_SR = 8000, 48000
MAX_SECONDS = float(os.environ.get('REUSE_MAX_SECONDS', str(3 * 3600)))

import json  # noqa: E402
CFG = json.load(open(os.path.join(MODEL_DIR, 'config.json')))
N_FFT, HOP, WIN = CFG['stft_cfg']['n_fft'], CFG['stft_cfg']['hop_size'], CFG['stft_cfg']['win_size']
COMPRESS, BASE_SR = CFG['model_cfg']['compress_factor'], CFG['stft_cfg']['sampling_rate']
DEV = torch.device('cuda')
MODEL = SEMamba.from_pretrained(MODEL_DIR, cfg=CFG).to(DEV).eval()
RELU = nn.ReLU()
GPU = torch.cuda.get_device_name(0)


def _even(v):
    v = int(round(v))
    return v if v % 2 == 0 else v + 1


@torch.no_grad()
def enhance(x, sr):
    """One channel through RE-USE, NVIDIA's inference_chunk.py recipe (Hann overlap-add of chunks)."""
    nf, hs, ws = _even(N_FFT * sr // BASE_SR), _even(HOP * sr // BASE_SR), _even(WIN * sr // BASE_SR)
    t = torch.tensor(x, dtype=torch.float32, device=DEV)[None]
    chunk = min(int(CHUNK_S * sr), t.shape[1])
    hl = max(1, int(HOP_P * chunk))
    w = torch.clamp(torch.hann_window(chunk, device=DEV), min=1e-7)
    out = torch.zeros_like(t)
    wsum = torch.zeros_like(t)
    n = max(1, math.ceil((t.shape[1] - chunk) / hl) + 1)
    for i in range(n):
        seg = t[:, i * hl: i * hl + chunk] if i < n - 1 else t[:, -chunk:]
        mag, pha, _ = mag_phase_stft(seg, n_fft=nf, hop_size=hs, win_size=ws, compress_factor=COMPRESS, center=True, addeps=False)
        amp, ph, _ = MODEL(mag, pha)
        m = torch.expm1(RELU(amp))
        zp = torch.sum(m == 0, 1) / m.shape[1]
        amp[:, :, (zp > 0.5)[0]] = 0          # NVIDIA's guard against the "strange sweep artifact"
        y = pad_or_trim_to_match(seg, mag_phase_istft(amp, ph, nf, hs, ws, COMPRESS), pad_value=1e-8)
        sl = slice(i * hl, i * hl + chunk) if i < n - 1 else slice(t.shape[1] - chunk, t.shape[1])
        out[:, sl] += y * w[:y.shape[1]]
        wsum[:, sl] += w[:y.shape[1]]
    return (out / wsum)[0].cpu().numpy()


def handler(job):
    inp = job.get('input') or {}
    if inp.get('probe'):
        a = torch.ones(64, 64, device=DEV)
        return {'ok': True, 'probe': True, 'gpu': GPU, 'sum': float((a @ a).sum())}
    src, dst = inp.get('source_url'), inp.get('result_url')
    if not src or not dst:
        return {'error': 'missing_urls'}
    deadline = float(inp.get('deadline_at') or 0)
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as td:
            sp = os.path.join(td, 'source')
            reuse_io.download(src, sp, deadline=deadline)
            try:
                x, sr = sf.read(sp, always_2d=True, dtype='float32')
            except Exception as e:
                return {'error': 'source_unreadable', 'detail': str(e)[:120]}
            if sr < MIN_SR or sr > MAX_SR:
                return {'error': 'bad_rate', 'sr': sr}
            if len(x) / sr > MAX_SECONDS:
                return {'error': 'too_long', 'seconds': round(len(x) / sr)}
            if x.shape[1] > 2:
                x = x[:, :2]
            # A stereo file whose channels are the same recording is cleaned once (half the GPU time).
            same = x.shape[1] == 2 and float(np.max(np.abs(x[:, 0] - x[:, 1]))) < 1e-4
            chans = [x[:, 0]] if (x.shape[1] == 1 or same) else [x[:, 0], x[:, 1]]
            if deadline and time.time() > deadline:
                return {'error': 'deadline_passed'}
            g0 = time.time()
            outs = [enhance(c, sr) for c in chans]
            torch.cuda.synchronize()
            gpu_s = time.time() - g0
            y = np.stack(outs * 2 if same else outs, 1)
            peak = float(np.max(np.abs(y))) if y.size else 0.0
            if peak > 0.999:
                y = y * (0.999 / peak)
            op = os.path.join(td, 'result.flac')
            sf.write(op, y, sr, format='FLAC', subtype='PCM_24')
            size = os.path.getsize(op)
            reuse_io.upload(dst, op, content_type='audio/flac', deadline=deadline)
            return {'ok': True, 'sr': sr, 'channels': int(x.shape[1]), 'processed_channels': len(chans),
                    'audio_s': round(len(x) / sr, 2), 'gpu_s': round(gpu_s, 2), 'wall_s': round(time.time() - t0, 2),
                    'bytes': size, 'gpu': GPU}
    except reuse_io.TransferError as e:
        return {'error': e.code, 'detail': str(e)[:160]}
    except torch.cuda.OutOfMemoryError:
        return {'error': 'gpu_oom'}


if __name__ == '__main__':
    import runpod
    runpod.serverless.start({'handler': handler})
