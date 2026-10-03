# Hub 모델을 Pipeline으로 실행하고, 셸 명령과 스킬로 연결하기

같은 추론 코드를 세 가지 방식으로 실행해 봅니다. 먼저 Python의 `pipeline()`으로 이미지를 분류하고, 이 코드를 `.sh`로 Colab GPU에서 실행합니다. 선택 실습에서는 그 명령을 AI 코딩 도구의 스킬로 연결합니다. 모델을 추가 학습하는 과정은 02번 노트북에서 이어갑니다.

## 1. `pipeline()`으로 Hub 모델 불러오기

Transformers의 함수 이름은 소문자 **`pipeline()`**입니다. 이미지 전처리, 모델 계산, 라벨·점수 정리를 하나의 호출로 묶어 줍니다. 아래 코드는 **라이브러리 설치를 마친 Colab GPU의 Python**에서 실행하는 예입니다. `input.png`는 해당 실행 환경에 준비한 이미지입니다.

```python
from PIL import Image
import torch
from transformers import AutoImageProcessor, pipeline

if not torch.cuda.is_available():
    raise RuntimeError("Colab GPU를 먼저 연결하세요.")

MODEL_ID = "facebook/deit-tiny-patch16-224"
MODEL_REVISION = "b3428f18dcc7b543470d07f14b4a4157815d1880"
processor = AutoImageProcessor.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, use_fast=False, trust_remote_code=False,
)
classifier = pipeline(
    task="image-classification",
    model=MODEL_ID,
    revision=MODEL_REVISION,
    image_processor=processor,
    framework="pt",
    device=0,
    trust_remote_code=False,
    model_kwargs={
        "weights_only": True,
        "use_safetensors": False,
        "attn_implementation": "eager",
    },
)
image = Image.open("input.png").convert("RGB")
for prediction in classifier(image, top_k=5):
    print(prediction["label"], f'{prediction["score"]:.1%}')
```

`model`에는 Hub의 저장소 ID를, `revision`에는 이번 수업의 고정 버전을 넣습니다. 모델 파일이 없다면 **이 Python 코드가 실행되는 컴퓨터로** 내려받습니다. 공개 모델이므로 HF 토큰은 필요하지 않습니다. `device=0`은 첫 번째 GPU입니다. `pipeline()` 자체가 원격 GPU를 만들어 주지는 않습니다. [Transformers Pipeline 공식 문서](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/pipelines)

이 revision의 원본 가중치는 `pytorch_model.bin`입니다. 따라서 위 코드는 `.bin` 파일을 읽도록 명시하고, `weights_only=True`로 로딩 범위를 제한합니다. 이 옵션이나 고정 revision이 임의의 모델 파일을 모두 안전하게 만들어 주는 것은 아닙니다.

수업의 Transformers **4.57.6**에서 제공된 추론 함수와 비교할 수 있도록 전처리는 `use_fast=False`, Attention 계산 방식은 `eager`로 맞췄습니다. Python 파일의 `create_classifier()`도 같은 설정을 사용합니다.

| 방법 | 파일 준비와 추론 | 수업에서 확인할 점 |
|---|---|---|
| `hf download` → `from_pretrained` | 파일을 먼저 내려받고, 전처리·모델 호출을 직접 작성 | 00·01번에서 실행 단계를 나눠 관찰합니다. |
| Hub ID를 넘긴 `pipeline()` | 필요한 모델을 받고 전처리·추론·라벨 정리를 연결 | 적은 코드로 원본 모델의 예측을 확인합니다. |

두 방식 모두 Transformers·PyTorch로 모델을 실행합니다. 원본 모델의 출력은 **ImageNet 1,000개 라벨**이며, 02번에서 학습할 병·그릇·캔·컵·접시 다섯 라벨과 다릅니다. `score`도 정답일 확률로 단정하지 않습니다.

01번에는 Pipeline 예제가 함께 들어 있습니다. 이어지는 두 함수에는 Pipeline이 묶어 준 처리 중 전처리·추론과 Top-k 정리 코드가 들어 있습니다. 같은 입력으로 두 방식의 결과를 비교하고 각 단계를 읽어 봅니다.

## 2. `.sh` 한 번으로 Colab에서 시연하기

실행 파일은 [run_pipeline_colab.sh](run_pipeline_colab.sh), 실제 추론 코드는 [pipeline_inference.py](pipeline_inference.py)입니다. 셸 파일을 열면 Google 인증 확인, GPU 생성, 파일 전송, 실행, 결과 회수, 종료에 해당하는 CLI 호출을 읽을 수 있습니다. Python에서는 `torch`, `PIL`, `transformers`를 직접 불러옵니다.

아래 명령은 **Codespaces의 프로젝트 최상위 터미널**에서 실행합니다. 자동 환경 설치가 끝나고 `.venv`가 준비되어 있어야 합니다.

```bash
source .venv/bin/activate

# 실행 계획만 확인: 로그인·이미지 준비·GPU 생성 없이 끝납니다.
bash hf_colab_gpu/run_pipeline_colab.sh --dry-run

# 실제 시연 전에 터미널에서 본인 Google 로그인을 마칩니다.
colab --auth oauth2 sessions

# 실제 시연: 00번이 준비한 테스트 데이터의 첫 cup 이미지를 사용합니다.
bash hf_colab_gpu/run_pipeline_colab.sh
```

기본 이미지를 쓰려면 먼저 00번을 끝내 `data/prepared/test.npz`와 `manifest.json`을 준비합니다. 본인 이미지를 쓰는 경우에는 00번 데이터 대신 파일 경로를 넘길 수 있습니다.

```bash
bash hf_colab_gpu/run_pipeline_colab.sh --image custom_images/my-cup.png
```

실제 실행은 Google 인증을 확인한 뒤 **새 T4 세션 하나**를 만듭니다. 기본 인증 방식은 `oauth2`입니다. 래퍼는 원격 작업 로그를 비공개 로컬 파일에 저장하므로, 첫 로그인은 위의 `colab --auth oauth2 sessions`를 **직접 실행해** 터미널 안내를 보고 완료합니다. CLI 0.6.0에는 `colab login` 명령이 없습니다. Codespaces 사용량과 Colab CU는 별개입니다.

이미 ADC 인증을 준비한 강사는 같은 인증 방식을 끝까지 사용합니다.

```bash
colab --auth adc sessions
bash hf_colab_gpu/run_pipeline_colab.sh --auth adc
```

| 옵션 | 쓰는 경우 |
|---|---|
| `--dry-run` | 어떤 파일과 명령을 사용할지 확인할 때. 원격 연산은 하지 않습니다. |
| `--image 경로` | 준비된 PNG·JPEG 등 본인 이미지를 사용할 때 |
| `--output-dir 경로` | 결과를 둘 새 폴더를 직접 정할 때 |
| `--session 이름` | 생성할 새 세션 이름을 지정할 때. 기존 세션을 재사용하지 않습니다. |
| `--auth oauth2` 또는 `--auth adc` | Google 인증 방식을 선택할 때 |

순서는 **새 T4 생성 → requirements 설치 → 이미지·Python 전송 → Colab에서 Hub 모델 다운로드와 추론 → JSON 회수·검사 → 이번 세션 종료 확인**입니다. 기본 결과 위치는 `hf_colab_gpu/results/pipeline` 아래 실행별 폴더입니다. 실제 경로는 마지막 안내에서 확인합니다.

자동 시연은 완료·오류·사용자 중단 때 이번에 만든 세션의 종료를 시도합니다. 결과 회수에 실패해도 종료하며, 남은 로그와 원격 경로를 출력합니다. 종료된 런타임의 파일은 다시 회수하지 못할 수 있습니다. 네트워크 단절 등으로 종료 확인에 실패했다면 안내된 세션만 확인하고 정리합니다. 다른 작업의 세션은 종료하지 않습니다. 터미널 프로세스를 강제로 없애는 상황까지 자동 종료를 보장할 수는 없습니다.

## 3. 출력을 읽고 조건 바꾸기

성공 여부는 마지막 문장만 보고 판단하지 않습니다. 이번 실행에서 회수한 JSON을 열어 다음 항목을 함께 읽습니다. 실제 파일 이름과 로그는 실행 종료 안내에 표시됩니다.

| JSON 항목 | 확인할 내용 |
|---|---|
| `status`, `run_id` | `completed`인지, 이번 실행 ID와 같은지 |
| `model_id`, `model_revision` | 어떤 Hub 모델의 어느 버전인지 |
| `device`, `device_name` | `cuda`와 실제 GPU 이름이 기록됐는지 |
| `image_sha256` | 이번 입력 이미지의 파일 지문인지 |
| `predictions` | 높은 점수부터 정렬된 `label`·`score` 목록인지 |
| `versions` | 실제 사용한 라이브러리 버전이 무엇인지 |

예상되는 결과의 **형태**는 `label`과 `score`를 가진 예측 5개입니다. 라벨과 점수는 사진에 따라 달라집니다. [2026-09-21 실제 T4 셸 시연](pipeline-demo-results.md)에 사용한 이미지, 실제 예측과 세션 종료 기록이 있습니다. [기본 학습 검증 기록](test-results.md)과 실행 날짜·범위를 구분해 읽습니다.

다음 중 하나를 시도한 뒤 달라진 점을 설명합니다.

1. 같은 물체를 밝은 배경과 어두운 배경에서 찍으면 상위 예측이 달라지는지 비교합니다.
2. 01번 Pipeline 셀의 `top_k`를 5에서 3으로 바꿉니다. 모델을 다시 학습했는지, 보여 주는 후보 수만 달라졌는지 설명합니다.
3. Pipeline 결과와 01번에 제공된 추론 함수의 결과를 같은 이미지에서 비교합니다. 모델·전처리·입력·장치가 같았는지 먼저 확인합니다.

명령행 Python 예제를 GPU가 설치된 별도 환경에서 직접 실행할 수도 있습니다.

```bash
python hf_colab_gpu/pipeline_inference.py \
  --image input.png \
  --output hf_colab_gpu/results/pipeline/manual/report.json \
  --device cuda --run-id manual-demo --top-k 5
```

이 명령은 **명령을 실행한 컴퓨터에서** 추론합니다. 기본 Codespaces 환경에는 GPU용 PyTorch를 설치하지 않으므로 Colab 시연에는 앞의 `.sh`를 사용합니다. 이미 필요한 라이브러리가 있는 CPU 환경에서만 `--device cpu`로 명시해 비교할 수 있습니다. CPU 실행 기록을 Colab GPU 검증 결과로 쓰지는 않습니다.

## 4. 선택 실습 · 추론 명령을 스킬로 등록하기

스킬은 AI 코딩 도구가 읽는 작업 안내입니다. 새 Python 패키지나 추론 서버가 아닙니다. 이 예제는 **스킬 → 셸 파일 → Colab CLI → Python Pipeline** 순서로 기존 코드를 사용합니다.

저장소에는 다음 두 파일이 들어 있습니다.

```text
.agents/skills/vision-pipeline-inference/
├── SKILL.md
└── agents/openai.yaml
```

[SKILL.md](../.agents/skills/vision-pipeline-inference/SKILL.md)에는 언제 사용할지, 어떤 명령을 실행할지, 무엇을 결과로 확인할지가 들어 있습니다. `agents/openai.yaml`은 표시 이름과 시작 프롬프트입니다. 이 폴더를 함께 배포하는 것이 이번 저장소의 등록 예시입니다.

Codex에서는 이 저장소를 작업 폴더로 열면 저장소의 `.agents/skills`를 검색합니다. 스킬이 목록에 나타나면 아래처럼 호출할 수 있습니다. 변경 후 목록에 보이지 않으면 Codex를 다시 시작하고 현재 연 폴더를 확인합니다. 도구별로 스킬 지원과 읽는 위치는 다를 수 있습니다. [공식 스킬 안내](https://learn.chatgpt.com/docs/build-skills)

**GPU를 만들지 않고 설명만 요청하는 예:**

> `$vision-pipeline-inference`로 이번 Pipeline 시연의 실행 계획만 보여 줘. dry-run으로 파일과 명령을 확인하고 GPU는 만들지 마.

**실제 실행을 요청하는 예:**

> `$vision-pipeline-inference`로 00번에서 준비한 cup 이미지를 새 Colab T4에서 추론해 줘. 결과 JSON을 회수하고 이번에 만든 세션이 종료됐는지도 확인해 줘.

인증이 필요하면 본인이 Google 로그인 절차를 마칩니다. 스킬 파일에 토큰을 적거나 계정을 공유하지 않습니다. 스킬을 읽거나 설명하는 것만으로 원격 자원을 만들지는 않습니다.

시연 후에는 01·02번에 제공된 핵심 함수를 읽고 [기본 실습 안내](README.md)의 수동 실행으로 이어갑니다. 이 자동 추론 시연과 00·01·02번 학습 과정은 결과 폴더와 세션을 나누어 사용합니다.
