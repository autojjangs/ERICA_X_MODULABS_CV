# Hugging Face CLI + Colab GPU 기본 실습

**모델을 받아서 추론하고, 우리 물체에 맞게 추가 학습합니다.** 모델 구조는 Hugging Face Transformers로 불러오고, PyTorch로 GPU 연산을 합니다. 실습 파일은 이 폴더 안의 00·01·02번 노트북 세 개입니다.

모든 명령은 `.vision-lab-root`가 있는 **프로젝트 최상위 폴더의 Codespaces 터미널**에서 실행합니다. `hf_colab_gpu` 안으로 이동하지 않습니다. 기본 설정은 Google에 로그인하거나 GPU를 할당하지 않습니다.

| 도구 | 이 수업에서 하는 일 | 실행 위치 |
|---|---|---|
| Hugging Face CLI의 `hf download` | 원본 모델과 전처리 설정 다운로드 | Codespaces |
| Colab CLI의 `colab` | GPU 생성·파일 전송·노트북 실행·회수·종료 | 명령은 Codespaces, 연산은 Colab |
| Transformers | `pipeline()` 추론, 모델 구조와 이미지 전처리 불러오기 | Colab GPU |
| PyTorch | GPU 추론, 손실·미분·가중치 업데이트 | Colab GPU |
| NumPy·Pillow·PyArrow | 이미지 읽기와 데이터 분할 | Codespaces |

## 1. 환경과 모델·데이터 준비

처음 만든 Codespace는 자동 설정을 마친 뒤 다음 명령으로 환경을 확인합니다. 이미 쓰던 환경이면 `bash scripts/setup.sh`를 한 번 실행해 HF CLI를 설치합니다.

```bash
source .venv/bin/activate
python scripts/doctor.py
```

처음 VS Code가 폴더 신뢰 여부를 물으면, 본인이 만든 강의 저장소인지 확인한 뒤 신뢰를 승인합니다. [00_hf_download_and_data.ipynb](notebooks/00_hf_download_and_data.ipynb)를 열고 **커널 선택 → Jupyter 커널 → Vision AI (Codespaces CPU)**를 선택한 뒤 **모두 실행(Run All)**을 누릅니다.

00번 앞부분에 Fork·Codespace 생성부터 devcontainer 설정 파일, 자동 설치, Python 경로, 커널 선택, 재빌드까지 안내했습니다. `devcontainer.json`이 `Dockerfile`과 `scripts/setup.sh`를 어떻게 연결하는지 읽고 첫 코드 셀로 넘어가세요.

이 노트북이 아래 공개 CLI를 실제로 호출합니다.

```bash
hf download facebook/deit-tiny-patch16-224 config.json preprocessor_config.json pytorch_model.bin \
  --revision b3428f18dcc7b543470d07f14b4a4157815d1880 \
  --local-dir hf_colab_gpu/models/deit-tiny
```

공개 모델이므로 Hugging Face 로그인은 필요하지 않습니다. 명령만 따로 실행했다면 00번의 파일 검사·다운로드 기록·데이터 준비 셀도 실행하세요. 모델은 약 23 MB, 처음 받는 데이터 원본은 약 142 MB입니다.

`hf download`는 모델 파일을 받는 명령입니다. 그 파일의 가중치를 Python 모델로 읽는 단계는 GPU 노트북의 `from_pretrained(...)`가 담당합니다. [고정 버전 CLI 문서](https://github.com/huggingface/huggingface_hub/blob/v0.36.0/docs/source/en/guides/cli.md)

준비가 끝나면 `hf_colab_gpu/models/deit-tiny`에 모델 3개 파일과 `download_manifest.json`, `data/prepared`에 데이터 3개 NPZ와 `manifest.json`이 생깁니다.

## 먼저 시연하기 · Hub 모델 Pipeline → 셸 파일

[Pipeline 시연 안내](pipeline-guide.md)의 `transformers.pipeline()`은 Hub의 모델 ID와 고정 revision으로 이미지를 분류합니다. Python 추론 코드를 `.sh`로 감싸 Colab GPU에서 실행하고 결과까지 회수해 봅니다.

```bash
# 로그인이나 GPU 생성 없이 명령과 경로를 먼저 확인합니다.
bash hf_colab_gpu/run_pipeline_colab.sh --dry-run

# 처음에는 터미널에서 본인 Google 로그인을 먼저 마칩니다.
colab --auth oauth2 sessions

# 새 T4에서 추론하고 이번에 만든 세션을 종료합니다.
bash hf_colab_gpu/run_pipeline_colab.sh
```

시연은 00번이 준비한 첫 cup 이미지와 별도 T4 세션을 사용합니다. `--image custom_images/my-cup.png`로 본인 이미지를 지정할 수도 있습니다. 이 자동 시연은 설치·실행·JSON 회수 후 종료까지 진행하며 실패한 경우에도 종료를 시도합니다. 뒤의 01·02번 수동 학습 세션은 별도로 만듭니다. 인증 방식, 로그, 오류 처리와 선택 스킬 예제는 [시연 안내](pipeline-guide.md)를 따릅니다.

## GPU 생성 전 · 제공된 코드 확인

01번과 02번의 핵심 함수는 **각각 두 개씩 완성된 코드로 제공**합니다. [핵심 코드와 조건 변경 실습](ai-coding-workshop.md)에서 함수의 입력·출력과 처리 순서를 먼저 확인하세요. 코드를 채우는 과정 없이 제공된 파일을 실행할 수 있으며, 실행 후에는 조건을 바꾸며 결과를 비교합니다.

```bash
python scripts/check_student_cells.py
```

GPU 할당 전에 제공된 함수와 Python 문법을 확인합니다. 코드를 수정했다면 노트북을 저장한 뒤 같은 검사를 다시 실행합니다. 이 사전 검사는 계산 결과를 보장하지 않으므로 GPU 실행 뒤 노트북의 확인 셀과 출력 결과도 읽습니다.

## 2. Google 로그인과 GPU 세션 생성

```bash
colab --auth oauth2 sessions
colab new -s hf-vision-gpu --gpu T4
colab status -s hf-vision-gpu
```

첫 명령의 안내에 따라 Google 로그인을 마칩니다. CLI 0.6.0에는 `colab login` 명령이 없습니다. 기존 ADC 인증을 쓰는 강사는 이후 명령에서도 `colab --auth adc ...`를 일관되게 사용합니다. 새 학생은 위 OAuth2 흐름을 사용하면 됩니다.

동일한 이름의 세션이 이미 있으면 새 이름을 정해 이후 명령에도 사용하세요. `new`부터 Colab 자원을 사용하므로 CU와 가용량을 먼저 확인합니다. Codespaces 사용료와 Colab CU는 별개입니다.

## 3. GPU 라이브러리 설치와 연결 확인

```bash
colab upload -s hf-vision-gpu hf_colab_gpu/requirements-torch.txt content/requirements-torch.txt
colab upload -s hf-vision-gpu hf_colab_gpu/requirements-gpu.txt content/requirements-gpu.txt
colab exec -s hf-vision-gpu --timeout 1800 <<'PY'
import subprocess
import sys

subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', '/content/requirements-torch.txt'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', '/content/requirements-gpu.txt'], check=True)
print('HF_GPU_LIBRARIES_READY')
PY
colab restart-kernel -s hf-vision-gpu
```

첫 파일은 공식 CUDA 12.6 배포처의 PyTorch·Torchvision 조합, 두 번째 파일은 Transformers·NumPy·Pillow 등 나머지 라이브러리 목록입니다. `colab install`도 설치 명령이지만 0.6.0에서는 내부 대기시간이 짧아 큰 PyTorch 설치가 중단될 수 있습니다. 위처럼 requirements를 보내고 `exec --timeout 1800`에서 공개 `pip` 명령을 실행합니다. `HF_GPU_LIBRARIES_READY`와 오류 유무를 확인하고, 설치 전에 메모리에 들어간 버전과 섞이지 않도록 커널을 다시 시작합니다. [PyTorch 공식 버전 조합](https://pytorch.org/get-started/previous-versions/)

다음 코드는 **현재 Colab 세션 안에서** 실행됩니다.

```bash
colab exec -s hf-vision-gpu --timeout 120 <<'PY'
import torch

assert torch.cuda.is_available(), 'GPU를 찾지 못했습니다.'
print('PyTorch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0))
PY
```

Codespaces에서 `import torch`를 실행하는 것과 다릅니다. CLI가 연결한 Colab GPU 커널에서 실행합니다. GPU가 없으면 CPU로 대체하지 않고 중단합니다.

## 4. 모델과 데이터를 파일별로 전송

`upload`는 상위 폴더를 자동 생성하지 않으므로 먼저 폴더를 만듭니다.

```bash
colab exec -s hf-vision-gpu --timeout 120 <<'PY'
from pathlib import Path

root = Path('/content/vision-ai')
(root / 'hf_colab_gpu/models/deit-tiny').mkdir(parents=True, exist_ok=True)
(root / 'data/prepared').mkdir(parents=True, exist_ok=True)
(root / 'hf_colab_gpu/results/gpu').mkdir(parents=True, exist_ok=True)
PY

colab upload -s hf-vision-gpu .vision-lab-root content/vision-ai/.vision-lab-root
colab upload -s hf-vision-gpu hf_colab_gpu/models/deit-tiny/config.json content/vision-ai/hf_colab_gpu/models/deit-tiny/config.json
colab upload -s hf-vision-gpu hf_colab_gpu/models/deit-tiny/preprocessor_config.json content/vision-ai/hf_colab_gpu/models/deit-tiny/preprocessor_config.json
colab upload -s hf-vision-gpu hf_colab_gpu/models/deit-tiny/pytorch_model.bin content/vision-ai/hf_colab_gpu/models/deit-tiny/pytorch_model.bin
colab upload -s hf-vision-gpu hf_colab_gpu/models/deit-tiny/download_manifest.json content/vision-ai/hf_colab_gpu/models/deit-tiny/download_manifest.json
colab upload -s hf-vision-gpu data/prepared/manifest.json content/vision-ai/data/prepared/manifest.json
colab upload -s hf-vision-gpu data/prepared/train.npz content/vision-ai/data/prepared/train.npz
colab upload -s hf-vision-gpu data/prepared/validation.npz content/vision-ai/data/prepared/validation.npz
colab upload -s hf-vision-gpu data/prepared/test.npz content/vision-ai/data/prepared/test.npz
```

`content/vision-ai/...`는 Colab 파일 API 경로이며 Python의 `/content/vision-ai/...`를 가리킵니다. 모델·데이터는 노트북 파일과 별도로 전송해야 합니다. Google 인증 정보나 HF 토큰 파일은 입력 파일에 포함하지 않습니다.

## 5. GPU 추론 → 파인튜닝

먼저 [01_gpu_inference.ipynb](notebooks/01_gpu_inference.ipynb)의 두 핵심 함수와 사전 검사 결과를 확인합니다. `AutoImageProcessor`, `AutoModelForImageClassification`, `torch.inference_mode`가 하는 일을 읽고 다음 CLI로 실행합니다.

```bash
colab exec -s hf-vision-gpu -f hf_colab_gpu/notebooks/01_gpu_inference.ipynb --timeout 1800
```

01번의 Hub 모델 `pipeline()` 예제와 제공된 추론 함수의 결과를 비교합니다. `01_gpu_inference_output.ipynb`의 모든 코드 셀에 오류가 없는지 확인한 뒤 다음 노트북으로 넘어갑니다. 기존 1,000개 ImageNet 라벨로 나온 Top-5는 관찰용입니다.

[02_gpu_finetuning.ipynb](notebooks/02_gpu_finetuning.ipynb)는 같은 원본 모델에서 새 5개 클래스 분류기를 학습한 뒤, 마지막 Transformer 블록·최종 정규화·분류기를 함께 학습합니다.

```bash
colab exec -s hf-vision-gpu -f hf_colab_gpu/notebooks/02_gpu_finetuning.ipynb --timeout 1800
```

PyTorch의 `loss.backward()`와 `optimizer.step()`을 학습 셀에서 직접 볼 수 있습니다. 모델 내부 Attention 연산을 재구현하지 않습니다. `from_pretrained`는 전송한 파일만 읽도록 `local_files_only=True`를 사용합니다. [Transformers 이미지 분류 안내](https://huggingface.co/docs/transformers/v4.57.1/en/tasks/image_classification)

CLI 0.6.0은 셀 오류가 나도 뒤의 셀을 실행할 수 있습니다. 명령 종료 코드만 보고 성공으로 판단하지 말고, `*_output.ipynb`에 오류가 없는지 확인합니다. `--timeout 1800`은 셀 대기시간이며 자동 세션 종료 시간이 아닙니다.

## 6. 결과 회수와 종료

회수할 구체적인 파일은 02번 마지막 셀과 보고서의 `artifacts`에서 확인합니다. 아래 예시는 기본 출력 경로를 사용합니다. 같은 이름의 로컬 결과가 있으면 먼저 다른 폴더에 보관하세요.

```bash
mkdir -p hf_colab_gpu/results/gpu/finetuned_model
colab download -s hf-vision-gpu content/vision-ai/hf_colab_gpu/results/gpu/inference_report.json hf_colab_gpu/results/gpu/inference_report.json
colab download -s hf-vision-gpu content/vision-ai/hf_colab_gpu/results/gpu/pretrained_top5.png hf_colab_gpu/results/gpu/pretrained_top5.png
colab download -s hf-vision-gpu content/vision-ai/hf_colab_gpu/results/gpu/report.json hf_colab_gpu/results/gpu/report.json
colab download -s hf-vision-gpu content/vision-ai/hf_colab_gpu/results/gpu/training.csv hf_colab_gpu/results/gpu/training.csv
colab download -s hf-vision-gpu content/vision-ai/hf_colab_gpu/results/gpu/predictions.csv hf_colab_gpu/results/gpu/predictions.csv
colab download -s hf-vision-gpu content/vision-ai/hf_colab_gpu/results/gpu/learning_curves.png hf_colab_gpu/results/gpu/learning_curves.png
colab download -s hf-vision-gpu content/vision-ai/hf_colab_gpu/results/gpu/confusion_comparison.png hf_colab_gpu/results/gpu/confusion_comparison.png
colab download -s hf-vision-gpu content/vision-ai/hf_colab_gpu/results/gpu/finetuned_model/config.json hf_colab_gpu/results/gpu/finetuned_model/config.json
colab download -s hf-vision-gpu content/vision-ai/hf_colab_gpu/results/gpu/finetuned_model/preprocessor_config.json hf_colab_gpu/results/gpu/finetuned_model/preprocessor_config.json
colab download -s hf-vision-gpu content/vision-ai/hf_colab_gpu/results/gpu/finetuned_model/model.safetensors hf_colab_gpu/results/gpu/finetuned_model/model.safetensors
colab stop -s hf-vision-gpu
colab sessions
```

출력 노트북은 Codespaces의 `hf_colab_gpu/notebooks/*_output.ipynb`에 저장됩니다. 보고서의 실제 장치·완료 상태·가중치 변경과 파일 회수를 확인하고 세션을 종료합니다. 준비나 학습에서 오류가 나도 새 세션을 반복 생성하지 말고 기존 세션 상태를 확인하세요. 복구를 마치거나 실습을 중단할 때 이번에 만든 세션을 종료합니다. 결과 다운로드만 실패했다면 파일이 남아 있는 같은 세션에서 회수부터 다시 시도합니다.

Colab 종료 뒤 결과 폴더와 출력 노트북을 학생 PC로 내려받고 Codespaces도 중지합니다.

## 수업 자료

- [00·01·02 코드 해설 노트북](explanations/README.md): 각 원본 코드 셀의 변수·문법·결과를 설명하고 CPU에서 작은 예제를 실행합니다.
- [Pipeline 추론·셸 시연·스킬 등록 예시](pipeline-guide.md)
- [실제 T4 셸 시연 결과와 종료 기록](pipeline-demo-results.md)
- [핵심 코드 네 가지와 조건 변경 실습](ai-coding-workshop.md)
- [학생 교안](handson.md)
- [실제 검증 결과](test-results.md)
- [새 GitHub Codespaces 생성·실행 검증 기록](codespaces-verification.md)
- [검증 결과를 포함한 최종 교안](final-guide.md)
- [양자화 선택 읽을거리](quantization-reference.md)

`validation`의 JSON·CSV·PNG는 과거 실행을 확인하는 참고 기록입니다. 실습은 `notebooks`의 세 파일로 진행하고, 직접 실행한 결과는 `results`와 출력 노트북에서 확인합니다.
