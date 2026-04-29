#!/usr/bin/env bash
# =====================================================================
# One-shot environment setup for the AI Video Pipeline.
#
# Run from the repo root or video_pipeline/ directory:
#     bash setup.sh
#
# Resolves all pinned dependencies in a single pip pass so transitive deps
# (notably TTS pulling in transformers) cannot upgrade past versions that
# break diffusers. If you've already broken your env by installing TTS
# unconstrained, this script will repin transformers + huggingface_hub
# back to the working set.
# =====================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

if [[ ! -f "${REQ_FILE}" ]]; then
    echo "ERROR: ${REQ_FILE} not found. Run from inside video_pipeline/." >&2
    exit 1
fi

echo "==> Upgrading pip"
python -m pip install --upgrade pip

echo "==> Installing pinned requirements (single resolve)"
pip install -r "${REQ_FILE}"

echo "==> Force-pinning critical versions in case a transitive dep upgraded them"
pip install --force-reinstall --no-deps \
    "transformers==4.57.6" \
    "huggingface_hub==0.36.2" \
    "diffusers==0.29.2" \
    "torch==2.11.0" \
    "torchvision==0.26.0"

echo "==> Verifying imports"
python - <<'PY'
import importlib, sys
mods = [
    "torch", "torchvision",
    "diffusers", "transformers", "huggingface_hub", "accelerate",
    "PIL", "imageio_ffmpeg",
    "anthropic",
]
fail = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  ok   {m}")
    except Exception as e:
        fail.append((m, e))
        print(f"  FAIL {m}: {e}")
try:
    from diffusers import AutoPipelineForText2Image  # noqa
    print("  ok   diffusers.AutoPipelineForText2Image")
except Exception as e:
    fail.append(("diffusers.AutoPipelineForText2Image", e))
    print(f"  FAIL diffusers.AutoPipelineForText2Image: {e}")
try:
    from TTS.api import TTS  # noqa
    print("  ok   TTS.api")
except Exception as e:
    fail.append(("TTS.api", e))
    print(f"  FAIL TTS.api: {e}")
sys.exit(1 if fail else 0)
PY

echo
echo "==> Setup complete. Run the pipeline with:"
echo "    python main.py --niche \"your topic here\" --no-approval"
