# 사전학습 모델을 받아 GPU에서 추론하고 파인튜닝하기

오늘 만들 것은 병·그릇·캔·컵·접시를 구분하는 이미지 분류기입니다. Physical AI의 앞단에서 물체 종류를 알려 주는 데 쓸 수 있지만, 이번 실습에서는 로봇 제어와 집기 좌표 계산까지 다루지 않습니다.

모델 구조는 Transformers가 불러옵니다. 학생이 집중할 부분은 **어떤 모델과 데이터를 받았는지, GPU에서 어떻게 실행하는지, 추가 학습으로 무엇이 달라지는지**입니다.

## 진행 순서

| 시간 | 활동 | 확인할 내용 |
|---|---|---|
| 0–35분 | Colab CLI·HF CLI·모델·가속기 소개 | 어디서 파일을 받고 어디서 연산하는가 |
| 35–55분 | Codespaces·devcontainer 준비와 00번 | Python·커널 경로, 모델 3개 파일, 데이터 500/100/200장 |
| 55–70분 | Hub Pipeline과 셸 파일 시연 | 새 Colab T4에서 추론·결과 회수·종료 |
| 70–100분 | Colab GPU 연결·전송·01번 추론 | Pipeline과 제공된 함수의 결과 비교 |
| 100–150분 | 02번 분류기 학습·파인튜닝 | 같은 테스트 이미지의 예측 변화 |
| 150–170분 | 저장·복원·결과 회수·종료 | 모델 재로딩과 GPU 세션 종료 |

처음 설치하거나 자원을 기다리는 시간은 계정·네트워크에 따라 달라집니다. 위 표는 실습 운영 예시입니다. 1시간 발표자료를 함께 쓰는 경우 소개 시간을 60분으로 잡고 뒤의 일정을 조정합니다.

## 두 CLI와 두 실행 환경 구분하기

`hf download`는 Hugging Face Hub에서 모델 파일을 내려받습니다. `colab`은 Colab 런타임을 만들고 그 안에서 Python 코드를 실행합니다. 하나는 파일을 받는 도구이고 다른 하나는 원격 컴퓨터를 다루는 도구입니다.

Codespaces에서는 노트북 편집·모델 다운로드·데이터 준비를 합니다. GPU 추론과 학습은 Colab에서 수행합니다. Colab GPU를 만들었다고 Codespaces의 Python 커널이 GPU 커널로 바뀌지는 않습니다. `colab exec -f`로 노트북의 코드 셀을 Colab에 보내야 합니다.

Colab CU와 Codespaces 사용료는 별개입니다. Colab CU 잔액·할당 가능 여부를 확인하고, 실습 후 직접 만든 세션을 종료합니다. Runpod이나 GCP와 비용 구조를 비교할 때에도 ‘과금이 절대 발생하지 않는다’고 설명하지 않습니다. [Colab FAQ](https://research.google.com/colaboratory/faq.html)

## 00 · 모델 다운로드와 데이터 준비

노트북 맨 앞의 devcontainer 안내부터 읽습니다. Fork한 저장소의 `.devcontainer/devcontainer.json`은 Dockerfile·설치 명령·확장을 연결하고, Dockerfile은 Python 3.12와 `uv`를 준비합니다. 자동으로 실행되는 `scripts/setup.sh`가 `.venv`와 `Vision AI (Codespaces CPU)` 커널을 만듭니다. 터미널에서 `python scripts/doctor.py`를 실행한 뒤 노트북도 같은 가상환경의 커널을 사용하는지 확인합니다.

`hf download`의 네 요소를 확인합니다. 모델 ID, 받을 파일 이름, 고정 revision, 저장할 폴더입니다. revision을 고정하면 같은 모델 버전을 다시 받을 수 있습니다. 파일 SHA-256 검사까지 마친 뒤 다음 단계로 갑니다.

공개 모델 `facebook/deit-tiny-patch16-224`는 ImageNet 1,000개 클래스로 사전학습됐습니다. 이 모델은 Transformers의 ViT 분류 모델로 불러옵니다. 이번 고정 revision의 원본 가중치는 `pytorch_model.bin`이며, 학습 후 저장하는 파일은 Safetensors입니다. 형식이 다른 것은 다운로드 실패가 아닙니다.

데이터는 학습·검증·테스트로 나눕니다. 학습은 가중치 업데이트, 검증은 저장할 모델 선택, 테스트는 최종 성능 확인에 사용합니다. CIFAR-100의 원본 이미지는 32×32입니다. 모델 입력인 224×224로 늘려도 새로운 세부 정보가 생기지는 않습니다.

실행할 파일: [00_hf_download_and_data.ipynb](notebooks/00_hf_download_and_data.ipynb)

## 시연 · Pipeline으로 추론하고 `.sh`로 실행하기

Transformers의 `pipeline("image-classification", model=모델_ID, ...)`은 전처리·모델 계산·라벨 정리를 연결합니다. 01번의 추론 함수에 담긴 처리 흐름을 적은 코드로 먼저 확인합니다. 모델 ID를 넘기면 Hub에서 해당 모델을 내려받으며, 연산은 `pipeline()`을 실행한 컴퓨터에서 합니다.

프로젝트 최상위 Codespaces 터미널에서 실행합니다.

```bash
bash hf_colab_gpu/run_pipeline_colab.sh --dry-run
colab --auth oauth2 sessions
bash hf_colab_gpu/run_pipeline_colab.sh
```

첫 명령은 설명만 보여 줍니다. 두 번째 명령을 직접 실행해 본인 Google 로그인을 마친 뒤 마지막 명령으로 시연합니다. 새 T4에서 00번의 첫 cup 이미지를 추론한 뒤 JSON을 회수하고 종료합니다. 래퍼는 모델 준비·설치·파일 전송과 종료를 묶은 셸 코드입니다. 실제 추론은 `pipeline_inference.py`에서 공개 라이브러리를 직접 불러와 수행합니다.

결과의 모델·revision, 장치 이름, Top-5, 실행 ID와 세션 종료 확인을 읽습니다. 사진을 바꿔 실행한 뒤 상위 라벨과 점수가 어떻게 달라졌는지 설명합니다. 원본 모델은 1,000개 ImageNet 라벨을 출력하므로 우리 다섯 클래스의 정확도로 해석하지 않습니다.

선택 시연에서는 `.agents/skills/vision-pipeline-inference/SKILL.md`를 읽고 AI 코딩 도구에 `$vision-pipeline-inference`로 이 명령을 요청합니다. 스킬이 새 모델을 만드는 것이 아니라 기존 셸 명령을 실행한다는 연결을 확인합니다. [추론 코드·옵션·스킬 등록 설명](pipeline-guide.md)

## 네 개의 핵심 코드 셀 읽고 실행하기

01번의 추론·Top-k 함수, 02번의 한 배치 학습·학습 범위 선택 함수를 완성된 코드로 제공합니다. 각 함수의 입력·출력과 처리 순서를 읽고, 같은 노트북의 확인 셀과 실제 출력으로 결과를 설명합니다. 실행을 마친 뒤 조건을 하나씩 바꿔 봅니다. 자세한 순서는 [핵심 코드 실습 안내](ai-coding-workshop.md)에 있습니다.

GPU 세션을 만들기 전에 Codespaces에서 `python scripts/check_student_cells.py`를 실행합니다. 이 검사는 제공된 함수와 Python 문법을 확인하며 GPU 계산은 하지 않습니다. 코드를 수정한 경우에는 입력 노트북을 저장한 뒤 다시 검사합니다. AI에게 코드 설명이나 수정 도움을 요청할 수 있지만 필수 단계는 아닙니다.

## 01 · GPU 연결과 원본 추론

[단계별 실행 안내](README.md)에 따라 Google 로그인 → T4 생성 → 라이브러리 설치 → 커널 재시작 → 파일 전송을 진행합니다. Colab 커널에서 `torch.cuda.is_available()`과 장치 이름을 확인합니다.

노트북의 Pipeline 예제에서는 같은 Hub 모델의 Top-5를 먼저 봅니다. 이어지는 두 함수는 전처리·모델 계산과 Top-k 정리를 나누어 보여 줍니다. 같은 입력에서 Pipeline과 제공된 함수의 결과가 맞는지 확인하고, 다르다면 모델 버전·전처리·정렬 방식을 점검합니다.

모델을 불러오는 핵심 코드는 다음 두 호출입니다.

```python
processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
model = AutoModelForImageClassification.from_pretrained(model_dir, local_files_only=True)
```

실제 노트북에는 원본 가중치 형식과 GPU 지정도 명시돼 있습니다. `processor`는 이미지 크기와 정규화를 처리하고, `model`은 클래스 점수를 계산합니다. `torch.inference_mode()`는 추론에 필요 없는 기울기 계산을 끕니다.

Top-5를 보고 이미지와 맞는지 설명합니다. 원본 모델의 라벨 이름이 우리 다섯 클래스와 같지 않을 수 있습니다. 원본 1,000개 라벨의 출력과 새 5개 클래스의 정확도는 같은 기준이 아닙니다.

실행할 파일: [01_gpu_inference.ipynb](notebooks/01_gpu_inference.ipynb)

## 02 · 비교 기준을 학습하고 일부 가중치를 수정하기

먼저 기존 모델의 마지막 분류기를 5개 클래스 분류기로 바꿉니다. 새 분류기의 가중치는 무작위 상태이므로 그것을 곧바로 사전학습 성능이라고 부르지 않습니다. 백본을 고정하고 새 분류기만 3 epoch 학습한 결과를 비교 기준으로 사용합니다.

그다음 학습한 분류기에서 이어서 마지막 Transformer 블록·최종 LayerNorm·분류기를 3 epoch 더 학습합니다. 앞의 11개 블록은 고정합니다. 노트북에서 학습 대상으로 정한 파라미터를 출력하고, 학습 전후 파일 지문으로 실제 변경 범위를 검사합니다.

한 배치 학습 함수에서는 기존 기울기를 초기화하고, 예측과 정답의 차이를 손실로 만든 뒤, 기울기를 구해 가중치를 갱신하는 순서를 읽습니다. 이어 학습 범위 선택 함수에서는 분류기만 학습할 때와 마지막 블록까지 학습할 때의 대상 이름·개수를 비교합니다. 실제 두 번째 학습에는 `last_block`을 사용합니다. 모델 내부 Attention을 직접 구현할 필요는 없습니다.

저장할 epoch는 **검증 정확도가 높은 순서, 동률이면 검증 손실이 낮은 순서**로 고릅니다. 테스트 데이터는 이 선택에 사용하지 않습니다. 각 단계에서 선택을 마친 모델을 같은 테스트 200장으로 평가합니다.

실행할 파일: [02_gpu_finetuning.ipynb](notebooks/02_gpu_finetuning.ipynb)

## 결과를 해석하고 다시 사용하기

학습 곡선에서는 검증 성능이 어떻게 움직였는지 봅니다. 혼동행렬에서는 어떤 종류를 서로 헷갈렸는지 확인합니다. `predictions.csv`에서 ‘둘 다 정답’, ‘파인튜닝 후 정답’, ‘파인튜닝 후 오답’ 사례를 하나씩 고릅니다. 파인튜닝이 모든 클래스에서 개선되리라고 가정하지 않습니다.

`save_pretrained()`로 모델 설정·가중치·전처리 설정을 저장한 뒤 같은 폴더를 다시 불러옵니다. 노트북은 저장 전후 같은 이미지의 출력이 일치하는지 확인합니다. 실제 작업대 사진에 적용하려면 촬영 환경에 맞는 데이터를 따로 수집하고 평가해야 합니다.

학습 결과 파일과 출력 노트북을 회수한 뒤 Colab 세션을 종료합니다. 브라우저 창이나 Codespaces를 닫는 것만으로 Colab 세션 종료를 대신하지 않습니다.

## 제출물과 선택 읽기

제출할 자료는 실제 GPU 이름, 원본 추론 결과, 파인튜닝 전후 지표·오분류 사례, 저장 모델 재로딩 결과, 결과 회수와 세션 종료 기록입니다. 수업 예시의 수치를 본인의 실행 결과로 복사하지 않습니다.

[양자화 참고 자료](quantization-reference.md)는 수업 뒤 읽어 볼 자료입니다. 필수 실습이나 평가에 포함하지 않습니다. 이번 실습은 00·01·02번을 실행하고 결과를 회수한 뒤 세션을 종료하면 마칩니다.
