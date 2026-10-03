#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LAB_ROOT"
SKIP_STATE_LINK=0
for arg in "$@"; do
  case "$arg" in
    --skip-data) ;; # 이전 안내와의 호환용입니다. 데이터는 이제 00번 노트북에서 준비합니다.
    --skip-state-link) SKIP_STATE_LINK=1 ;;
    -h|--help)
      printf '%s\n' 'Usage: bash scripts/setup.sh [--skip-state-link]' \
        'Hugging Face CLI와 Colab CLI, CPU에서 모델·데이터를 준비할 라이브러리를 설치합니다.' \
        '데이터는 00번 노트북에서 직접 준비합니다. 로그인하거나 Colab 자원을 생성하지 않습니다.'
      exit 0 ;;
    *) printf 'Unknown option: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' 'uv is required. Rebuild the supplied dev container, which includes uv 0.12.9.' >&2
  exit 1
fi
export UV_CACHE_DIR="$LAB_ROOT/.cache/uv"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$LAB_ROOT/.cache/matplotlib"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
uv venv --python 3.12 --allow-existing .venv
uv pip install --python .venv/bin/python -r requirements/codespaces.txt
.venv/bin/python -m ipykernel install --prefix "$LAB_ROOT/.venv" \
  --name codespaces-vision-ai --display-name 'Vision AI (Codespaces CPU)'

if [[ "$SKIP_STATE_LINK" -eq 0 ]]; then
  .venv/bin/python scripts/doctor.py --configure-state-only
fi
.venv/bin/python scripts/doctor.py
printf '\n%s\n' '준비 완료: 노트북 커널로 Vision AI (Codespaces CPU)를 선택하세요.' \
  '기본 실습은 hf_colab_gpu/README.md와 hf_colab_gpu/notebooks/00_hf_download_and_data.ipynb에서 시작하세요.' \
  '추론·파인튜닝은 Colab GPU에서 Transformers와 PyTorch로 실행합니다.' \
  '이 설치 과정에서는 Google 로그인이나 Colab 자원 생성이 일어나지 않습니다.'
