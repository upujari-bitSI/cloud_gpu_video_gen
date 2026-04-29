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
    "transformers==4.45.2" \
    "huggingface_hub==0.25.2" \
    "diffusers==0.31.0" \
    "torch==2.11.0" \
    "torchvision==0.26.0" \
    "sentencepiece>=0.2.0" \
    "protobuf>=4.25.0"

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
    print("  ok   diffusers.AutoPipelineForText2Image (SDXL fallback)")
except Exception as e:
    fail.append(("diffusers.AutoPipelineForText2Image", e))
    print(f"  FAIL diffusers.AutoPipelineForText2Image: {e}")
try:
    from diffusers import FluxPipeline  # noqa
    print("  ok   diffusers.FluxPipeline (primary)")
except Exception as e:
    fail.append(("diffusers.FluxPipeline", e))
    print(f"  FAIL diffusers.FluxPipeline: {e}")
try:
    from TTS.api import TTS  # noqa
    print("  ok   TTS.api (Coqui fallback)")
except Exception as e:
    print(f"  warn TTS.api missing: {e} (Coqui fallback unavailable)")
try:
    from kokoro import KPipeline  # noqa
    print("  ok   kokoro (default narrator)")
except Exception as e:
    fail.append(("kokoro", e))
    print(f"  FAIL kokoro: {e}")
try:
    import streamlit  # noqa
    print(f"  ok   streamlit {streamlit.__version__}")
except Exception as e:
    print(f"  warn streamlit missing: {e} (dashboard unavailable)")
sys.exit(1 if fail else 0)
PY

echo
echo "==> Setup complete."
echo "    CLI run:    python main.py --niche \"your topic\" --no-approval"
echo "    Dashboard:  streamlit run streamlit_app.py"
