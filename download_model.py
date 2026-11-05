# Bake NVIDIA RE-USE (weights + its model code + LICENSE) into the image at one pinned revision.
import os
from huggingface_hub import snapshot_download

REPO = 'nvidia/NVIDIA-NemotronLabs-RE-USE'
REVISION = '31f82f2efad1095034492f236e8dac9239ba713b'   # 2026-10-22 release, measured 05.11.2026
DST = os.environ.get('MODEL_DIR', '/models/reuse')

snapshot_download(REPO, revision=REVISION, local_dir=DST,
                  allow_patterns=['model.safetensors', 'config.json', 'LICENSE', 'models/*.py', 'utils/*.py'])
for need in ('model.safetensors', 'config.json', 'LICENSE', 'models/generator_SEMamba_time_d4.py', 'utils/util.py'):
    p = os.path.join(DST, need)
    assert os.path.exists(p), 'missing ' + need
print('RE-USE baked at', DST, 'revision', REVISION)
