#!/usr/bin/env bash
# Run the readable Python pipeline example on one newly created Colab T4.
# Every Colab operation is visible below; credentials never go into run-config.json.
set -Eeuo pipefail
umask 077

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
AUTH=oauth2
IMAGE=
OUTPUT=
SESSION=
DRY_RUN=0
usage() {
    cat <<'USAGE'
사용법: bash hf_colab_gpu/run_pipeline_colab.sh [옵션]
  --image PATH       본인 이미지. 생략하면 00번의 첫 cup 테스트 이미지
  --output-dir PATH  결과를 저장할 새 폴더 (기존 폴더는 덮어쓰지 않음)
  --session NAME     새 세션 이름 (기존 세션 재사용 불가)
  --auth oauth2|adc  Google 인증 방식. 기본값 oauth2
  --dry-run         실행 계획만 표시. 파일 생성·로그인·GPU 사용 없음
  --help            이 안내 표시

실제 시연 전에는 colab --auth oauth2 sessions로 본인 Google 로그인을 마치세요.
실제 실행은 Colab CU를 사용하며, 결과 회수 뒤 이번 세션을 종료합니다.
PYTHON_BIN, COLAB_BIN 환경변수로 실행 파일을 지정할 수 있습니다.
USAGE
}
fail() { printf '오류: %s\n' "$*" >&2; exit 1; }
while (($#)); do
    case "$1" in
        --image|--output-dir|--session|--auth)
            (($# >= 2)) && [[ -n "$2" ]] || fail "$1 값이 필요합니다."
            case "$1" in
                --image) IMAGE=$2;; --output-dir) OUTPUT=$2;;
                --session) SESSION=$2;; --auth) AUTH=$2;;
            esac
            shift 2;;
        --dry-run) DRY_RUN=1; shift;;
        --help|-h) usage; exit 0;;
        *) fail "알 수 없는 옵션: $1";;
    esac
done
[[ "$AUTH" == oauth2 || "$AUTH" == adc ]] || fail '--auth는 oauth2 또는 adc입니다.'
[[ -z "$SESSION" || "$SESSION" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]] || fail '세션 이름은 영문·숫자로 시작하는 1~64자의 영문·숫자·밑줄·하이픈입니다.'
if ((DRY_RUN)); then
    cat <<PLAN
실행 계획 (실제로 실행하지 않음)
입력: ${IMAGE:-data/prepared/test.npz의 첫 cup 이미지 (00번 준비 필요)}
결과: ${OUTPUT:-hf_colab_gpu/results/pipeline/<고유 실행 ID>}
세션: ${SESSION:-pipeline-<고유 실행 ID>} / 새 T4 / 인증 $AUTH
1. colab --auth $AUTH sessions: 이름 충돌 확인
2. colab --auth $AUTH new -s <새 세션> --gpu T4
3. colab --auth $AUTH exec: 원격 폴더 생성
4. colab --auth $AUTH upload: requirements 2개·pipeline_inference.py·이미지·설정
5. colab --auth $AUTH exec --timeout 1800: 라이브러리 설치 후 완료 JSON 확인
6. colab --auth $AUTH restart-kernel: 설치한 라이브러리 적용
7. colab --auth $AUTH exec -f <로컬 실행 파일> --timeout 600: Hub 모델 추론
8. colab --auth $AUTH download: 결과 JSON 회수·내용 검사 (실패하면 한 번 재시도)
9. colab --auth $AUTH stop: 이번 세션만 종료 후 sessions에서 종료 확인
PLAN
    exit 0
fi

# Prefer the course environment without relying on the caller's activated shell.
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON_BIN="$ROOT/.venv/bin/python"; else PYTHON_BIN=python3; fi
fi
if [[ -z "${COLAB_BIN:-}" ]]; then
    if [[ -x "$ROOT/.venv/bin/colab" ]]; then COLAB_BIN="$ROOT/.venv/bin/colab"; else COLAB_BIN=colab; fi
fi
command -v "$PYTHON_BIN" >/dev/null || fail 'Python을 찾지 못했습니다. Codespaces 환경 설치를 확인하세요.'
command -v "$COLAB_BIN" >/dev/null || fail 'Colab CLI를 찾지 못했습니다. Codespaces 환경 설치를 확인하세요.'
for name in pipeline_inference.py requirements-torch.txt requirements-gpu.txt; do
    [[ -r "$ROOT/hf_colab_gpu/$name" ]] || fail "필요한 파일이 없습니다: hf_colab_gpu/$name"
done
if [[ -n "$IMAGE" ]]; then
    [[ -r "$IMAGE" ]] || fail "이미지를 읽을 수 없습니다: $IMAGE"
else
    [[ -r "$ROOT/data/prepared/test.npz" && -r "$ROOT/data/prepared/manifest.json" ]] || fail '00번을 먼저 실행하거나 --image로 이미지를 지정하세요.'
fi
RUN_ID=$("$PYTHON_BIN" -c 'import uuid; print(uuid.uuid4().hex)')
SESSION=${SESSION:-pipeline-$RUN_ID}
OUTPUT=${OUTPUT:-$ROOT/hf_colab_gpu/results/pipeline/$RUN_ID}
# Python rejects existing/symlink destinations and prepares an RGB PNG before allocation.
"$PYTHON_BIN" - "$ROOT" "$OUTPUT" "$IMAGE" "$RUN_ID" "$SESSION" <<'PY'
import hashlib, json, sys
from pathlib import Path
from PIL import Image
root, output, source, run_id, session = sys.argv[1:]
root, output = Path(root), Path(output).absolute()
if output.exists() or output.is_symlink():
    raise SystemExit('기존 경로를 보존합니다. 새 --output-dir를 지정하세요.')
if any(parent.is_symlink() for parent in output.parents):
    raise SystemExit('결과 경로에 심볼릭 링크가 포함되어 있습니다. 실제 폴더를 지정하세요.')
if source:
    with Image.open(source) as opened:
        image = opened.convert('RGB')
else:
    import numpy as np
    manifest = json.loads((root / 'data/prepared/manifest.json').read_text())
    label = manifest['classes'].index('cup')
    with np.load(root / 'data/prepared/test.npz', allow_pickle=False) as data:
        matches = np.flatnonzero(data['labels'] == label)
        if not len(matches):
            raise SystemExit('테스트 데이터에 cup 이미지가 없습니다.')
        image = Image.fromarray(data['images'][matches[0]]).convert('RGB')
output.mkdir(parents=True, exist_ok=False, mode=0o700)
(output / 'logs').mkdir(mode=0o700)
image.save(output / 'input.png')
config = {
    'run_id': run_id, 'session': session,
    'model_id': 'facebook/deit-tiny-patch16-224',
    'model_revision': 'b3428f18dcc7b543470d07f14b4a4157815d1880',
    'image_sha256': hashlib.sha256((output / 'input.png').read_bytes()).hexdigest(),
    'files': {name: hashlib.sha256((root / 'hf_colab_gpu' / name).read_bytes()).hexdigest()
              for name in ('pipeline_inference.py', 'requirements-torch.txt', 'requirements-gpu.txt')},
}
(output / 'run-config.json').write_text(json.dumps(config, indent=2) + '\n')
PY
OUTPUT=$(cd -- "$OUTPUT" && pwd)
REMOTE="content/vision-pipeline/$RUN_ID"
CLI=("$COLAB_BIN" --auth "$AUTH")

# The helper validates evidence; it does not operate Colab or run the model.
cat > "$OUTPUT/verify.py" <<'PY'
import importlib.util, json, re, sys
from pathlib import Path
mode, output, *args = sys.argv[1:]
root = Path(output)
config = json.loads((root / 'run-config.json').read_text())
def require(condition, message):
    if not condition:
        raise SystemExit(message)
if mode == 'sessions':
    text = Path(args[0]).read_text()
    rows = []
    for line in text.splitlines():
        match = re.match(r'^\[([^\]]+)\]\s+(\S+)\s*\|\s*Hardware:\s*([^|]+)\|\s*Variant:\s*(\S+)', line.strip())
        if match:
            name, endpoint, hardware, variant = match.groups()
            rows.append(dict(name=name, endpoint=endpoint, hardware=hardware.strip(), variant=variant))
    require(rows or 'No active sessions found on server' in text, '세션 목록을 해석하지 못했습니다. Colab CLI 0.6.0과 인증을 확인하세요.')
    print(json.dumps(rows))
elif mode == 'collision':
    rows = json.loads((root / 'before.json').read_text())
    require(not any(row['name'] == config['session'] for row in rows), '같은 이름의 세션이 이미 있습니다. --session에 새 이름을 지정하세요.')
elif mode == 'identify':
    before = json.loads((root / 'before.json').read_text())
    rows = json.loads(Path(args[0]).read_text())
    matches = [row for row in rows if row['name'] == config['session']]
    require(len(matches) == 1, '이번에 생성한 세션을 식별하지 못했습니다.')
    row = matches[0]
    require(not any(old['endpoint'] == row['endpoint'] for old in before), '기존 세션을 종료할 수 없습니다.')
    (root / 'owned-session.json').write_text(json.dumps(row, indent=2) + '\n')
    require(row['hardware'] == 'T4' and row['variant'] == 'GPU', '생성된 세션이 요청한 T4 GPU가 아닙니다.')
elif mode == 'absent':
    rows = json.loads(Path(args[0]).read_text())
    owned = json.loads((root / 'owned-session.json').read_text())
    require(not any(row['name'] == config['session'] or row['endpoint'] == owned['endpoint'] for row in rows), '이번 세션이 아직 목록에 있습니다.')
elif mode == 'install':
    report = json.loads((root / 'install-complete.json').read_text())
    require(report.get('status') == 'installed' and report.get('run_id') == config['run_id'], '설치 완료 기록이 이번 실행과 다릅니다.')
    require(report.get('files') == config['files'], '원격 코드 또는 requirements가 원본과 다릅니다.')
    require(report.get('versions') == {'torch': '2.8.0+cu126', 'torchvision': '0.23.0+cu126', 'transformers': '4.57.6', 'huggingface-hub': '0.36.0'}, '설치 버전이 수업 버전과 다릅니다.')
elif mode == 'result':
    report = json.loads((root / 'result.json').read_text())
    for key in ('run_id', 'model_id', 'model_revision', 'image_sha256'):
        require(report.get(key) == config[key], f'추론 결과 {key}가 이번 실행과 다릅니다.')
    require(report.get('status') == 'completed' and report.get('task') == 'image-classification', '완료된 이미지 분류 결과가 아닙니다.')
    require(report.get('device') == 'cuda' and 'T4' in report.get('device_name', ''), 'T4 GPU 실행 결과가 아닙니다.')
    require(report.get('versions') == {'torch': '2.8.0+cu126', 'transformers': '4.57.6', 'huggingface-hub': '0.36.0'}, '추론 버전이 수업 버전과 다릅니다.')
    spec = importlib.util.spec_from_file_location('pipeline_inference', args[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_predictions(report.get('predictions'), 5)
elif mode == 'cleanup':
    (root / 'cleanup.json').write_text(json.dumps({
        'run_id': config['run_id'], 'session': config['session'],
        'termination_status': args[0], 'result_verified': args[1] == '1',
        'success': args[0] == 'confirmed' and args[1] == '1' and args[2] == '0',
    }, indent=2) + '\n')
PY
verify() { "$PYTHON_BIN" "$OUTPUT/verify.py" "$1" "$OUTPUT" "${@:2}"; }
# Suppress raw remote logs on the projector; keep them privately for troubleshooting.
step() {
    local label=$1
    shift
    "$@" >"$OUTPUT/logs/$label.log" 2>&1 || { printf '단계 실패: %s (비공개 로그: %s/logs)\n' "$label" "$OUTPUT" >&2; return 1; }
}
snapshot() {
    local label=$1
    step "$label" "${CLI[@]}" sessions || return 1
    verify sessions "$OUTPUT/logs/$label.log" >"$OUTPUT/$label.json"
}
CREATION_ATTEMPTED=0
RESULT_VERIFIED=0
cleanup() {
    local result=$? termination=not_started
    trap - EXIT INT TERM
    set +e
    if ((CREATION_ATTEMPTED)); then
        termination=unconfirmed
        if [[ ! -f "$OUTPUT/owned-session.json" ]]; then
            if snapshot cleanup-identify; then
                verify identify "$OUTPUT/cleanup-identify.json"
            fi
        fi
        if [[ -f "$OUTPUT/owned-session.json" ]]; then
            printf '이번에 만든 세션을 종료하고 확인합니다: %s\n' "$SESSION"
            step stop "${CLI[@]}" stop -s "$SESSION"
            local stop_status=$?
            if snapshot after-stop && verify absent "$OUTPUT/after-stop.json" && ((stop_status == 0)); then
                termination=confirmed
            fi
        fi
        if [[ "$termination" != confirmed ]]; then
            result=1
            printf '종료 확인 실패. 다른 세션을 종료하지 말고 이번 세션 %s만 확인하세요.\n' "$SESSION" >&2
            printf '확인 명령: %q --auth %q sessions\n' "$COLAB_BIN" "$AUTH" >&2
        fi
    fi
    verify cleanup "$termination" "$RESULT_VERIFIED" "$result" || result=1
    if ((result == 0 && RESULT_VERIFIED == 1)); then
        printf '시연 완료: T4 추론 결과와 세션 종료를 모두 확인했습니다.\n결과: %s/result.json\n종료 기록: %s/cleanup.json\n' "$OUTPUT" "$OUTPUT"
    elif ((result != 0)); then
        printf '시연이 완료되지 않았습니다. 로컬 기록: %s\n원격 작업 경로: /%s\n' "$OUTPUT" "$REMOTE" >&2
    fi
    exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf '1/6 인증과 세션 이름을 확인합니다. 첫 로그인은 colab --auth %s sessions로 먼저 마치세요.\n' "$AUTH"
snapshot before || fail '세션 목록을 읽지 못했습니다. 안내한 인증 명령을 직접 실행한 뒤 다시 시도하세요.'
verify collision
printf '2/6 새 T4 세션을 만듭니다: %s\n' "$SESSION"
CREATION_ATTEMPTED=1
step create "${CLI[@]}" new -s "$SESSION" --gpu T4
snapshot after-create
verify identify "$OUTPUT/after-create.json"

# `exec -f` reads local file contents, so do not depend on __file__ or CLI argv there.
cat > "$OUTPUT/mkdir-remote.py" <<PY
from pathlib import Path
Path('/$REMOTE').mkdir(parents=True, exist_ok=False)
print('REMOTE_DIRECTORY_READY')
PY
step mkdir "${CLI[@]}" exec -s "$SESSION" -f "$OUTPUT/mkdir-remote.py" --timeout 120
printf '3/6 코드와 이미지를 전송합니다.\n'
for name in pipeline_inference.py requirements-torch.txt requirements-gpu.txt; do
    step "upload-$name" "${CLI[@]}" upload -s "$SESSION" "$ROOT/hf_colab_gpu/$name" "$REMOTE/$name"
done
for name in input.png run-config.json; do
    step "upload-$name" "${CLI[@]}" upload -s "$SESSION" "$OUTPUT/$name" "$REMOTE/$name"
done
cat > "$OUTPUT/install.py" <<PY
import hashlib, json, subprocess, sys
from importlib.metadata import version
from pathlib import Path
root = Path('/$REMOTE')
config = json.loads((root / 'run-config.json').read_text())
assert config['run_id'] == '$RUN_ID'
for name, expected in config['files'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
for name in ('requirements-torch.txt', 'requirements-gpu.txt'):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(root / name)], check=True)
report = {'status': 'installed', 'run_id': config['run_id'], 'files': config['files'],
          'versions': {name: version(name) for name in ('torch', 'torchvision', 'transformers', 'huggingface-hub')}}
(root / 'install-complete.json').write_text(json.dumps(report, indent=2))
print('PIPELINE_INSTALL_COMPLETE')
PY
printf '4/6 GPU 라이브러리를 설치합니다. 몇 분 걸릴 수 있습니다.\n'
step install "${CLI[@]}" exec -s "$SESSION" -f "$OUTPUT/install.py" --timeout 1800
step install-receipt "${CLI[@]}" download -s "$SESSION" "$REMOTE/install-complete.json" "$OUTPUT/install-complete.json"
verify install
step restart "${CLI[@]}" restart-kernel -s "$SESSION"
cat > "$OUTPUT/run.py" <<PY
import hashlib, json, runpy, sys
from pathlib import Path
root = Path('/$REMOTE')
config = json.loads((root / 'run-config.json').read_text())
assert config['run_id'] == '$RUN_ID'
assert hashlib.sha256((root / 'pipeline_inference.py').read_bytes()).hexdigest() == config['files']['pipeline_inference.py']
assert hashlib.sha256((root / 'input.png').read_bytes()).hexdigest() == config['image_sha256']
sys.argv = ['pipeline_inference.py', '--image', str(root / 'input.png'),
            '--output', str(root / 'result.json'), '--device', 'cuda', '--run-id', config['run_id'],
            '--model-id', config['model_id'], '--revision', config['model_revision'], '--top-k', '5']
runpy.run_path(str(root / 'pipeline_inference.py'), run_name='__main__')
PY
printf '5/6 Hub 모델을 불러와 이미지를 추론합니다.\n'
step inference "${CLI[@]}" exec -s "$SESSION" -f "$OUTPUT/run.py" --timeout 600
printf '6/6 결과를 회수하고 검사합니다.\n'
if ! step download "${CLI[@]}" download -s "$SESSION" "$REMOTE/result.json" "$OUTPUT/result.json"; then
    printf '결과 회수를 한 번 더 시도합니다.\n'
    step download-retry "${CLI[@]}" download -s "$SESSION" "$REMOTE/result.json" "$OUTPUT/result.json"
fi
verify result "$ROOT/hf_colab_gpu/pipeline_inference.py"
RESULT_VERIFIED=1
# EXIT trap stops this owned session and prints success only after absence is verified.
