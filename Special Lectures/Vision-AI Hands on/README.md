# Vision AI · Hugging Face CLI와 Colab GPU

이 저장소의 실습은 **`hf_colab_gpu/notebooks`의 00·01·02번 세 개**입니다. **모델 다운로드 → GPU 추론 → 우리 물체로 파인튜닝 → 결과 확인** 순서로 진행합니다. 모델 구조는 Transformers로 불러오고, PyTorch로 학습합니다.

**[실습 시작하기 →](hf_colab_gpu/README.md)**

## 처음 시작하는 학생

1. 이 저장소의 **[Fork](https://github.com/yblee110/vision-ai-notebook/fork)**를 눌러 본인 GitHub 계정으로 복사합니다.
2. **본인이 Fork한 저장소**에서 **Code → Codespaces → Create codespace on main**을 선택합니다. 머신을 고르는 경우 **2코어·8GB**를 사용합니다.
3. 자동 환경 설치가 끝날 때까지 기다린 뒤 아래 명령으로 준비 상태를 확인합니다.
4. 기본 노트북 00번을 열고 **커널 선택 → Jupyter 커널 → Vision AI (Codespaces CPU)**를 선택해 실행합니다.

00번 앞부분에 **devcontainer 파일별 역할, 자동 설치, Python·커널 확인, 컨테이너 재빌드** 안내가 있습니다. 환경 준비가 낯설다면 코드 셀보다 이 안내를 먼저 읽습니다.

`.devcontainer`와 설치 설정은 Fork에 함께 포함됩니다. 기본 환경은 HF CLI·Colab CLI와 데이터 준비 라이브러리만 설치합니다. Google 로그인과 Colab GPU 할당은 학생이 본인 계정으로 실습 중 진행합니다. Codespaces와 Colab의 사용량은 별도로 관리합니다.

```bash
source .venv/bin/activate
python scripts/doctor.py
```

이미 쓰던 환경이라면 `bash scripts/setup.sh`로 필요한 라이브러리를 추가합니다. 가상환경은 하나를 사용하며, 프로젝트 자체를 설치 패키지로 묶지 않습니다. 코드 셀에서 공개 라이브러리를 직접 불러옵니다.

## 실습 노트북 3개

| 노트북 | 실행 위치 | 하는 일 |
|---|---|---|
| [00 · HF CLI와 데이터 준비](hf_colab_gpu/notebooks/00_hf_download_and_data.ipynb) | Codespaces CPU | 실제 `hf download`, 모델 파일 확인, 이미지 분할 |
| [01 · GPU 추론](hf_colab_gpu/notebooks/01_gpu_inference.ipynb) | Colab GPU | Hub 모델 `pipeline()`, 제공된 추론 함수, Top-5 비교 |
| [02 · GPU 파인튜닝](hf_colab_gpu/notebooks/02_gpu_finetuning.ipynb) | Colab GPU | 분류기 학습, 마지막 블록 파인튜닝, 비교·저장·재로딩 |

01·02의 핵심 함수 네 개는 **완성된 코드로 제공**합니다. [핵심 코드와 조건 변경 실습](hf_colab_gpu/ai-coding-workshop.md)을 읽고 GPU를 만들기 전에 `python scripts/check_student_cells.py`로 코드 문법과 필수 함수를 확인합니다. 먼저 제공된 코드로 실행한 뒤, 같은 노트북에서 조건을 바꾸고 결과를 비교합니다.

01·02는 Codespaces에서 편집하고 공식 `colab exec -f ...ipynb`로 실행합니다. GPU 연결·설치·입력 전송을 먼저 해야 하므로 [단계별 실행 안내](hf_colab_gpu/README.md)를 따라가세요. Codespaces CPU의 Run All은 Colab GPU에 자동으로 연결되지 않습니다.

## 코드 해설 함께 읽기

코드가 처음이라면 [00·01·02 코드 해설 노트북](hf_colab_gpu/explanations/README.md)을 실습 파일 옆에 두고 읽어 보세요. 원본 코드의 변수와 처리 순서를 풀어 설명하고, GPU 없이 실행하는 작은 예제를 제공합니다. 실습용 세 파일과 해설용 세 파일은 폴더를 나누었습니다.

## 짧은 시연 · Pipeline 추론을 셸 파일로 실행하기

00번을 마치면 같은 Hub 모델을 불러와 이미지 한 장을 추론하는 시연을 먼저 할 수 있습니다. 프로젝트 최상위 터미널에서 실행합니다.

```bash
# 로그인·GPU 생성 없이 실행 계획만 확인합니다.
bash hf_colab_gpu/run_pipeline_colab.sh --dry-run

# 처음에는 이 명령을 직접 실행해 본인 Google 로그인을 마칩니다.
colab --auth oauth2 sessions

# 실제 시연: 새 Colab T4 → Hub 모델 추론 → 결과 회수 → 세션 종료
bash hf_colab_gpu/run_pipeline_colab.sh
```

기본 입력은 00번이 준비한 테스트 데이터의 첫 cup 이미지입니다. 본인 사진은 `--image custom_images/my-cup.png`로 지정합니다. 실제 실행에는 본인 Google 인증과 Colab 자원이 필요하며 결과는 실행별 폴더에 저장됩니다. 기존 학습 세션은 재사용하지 않습니다.

[Pipeline 코드·셸 시연·선택 스킬 등록 안내](hf_colab_gpu/pipeline-guide.md)에서 각 명령의 역할과 출력 확인 방법을 읽습니다. 스킬 예제는 `.agents/skills/vision-pipeline-inference`에 있으며 같은 셸 파일을 호출합니다. 01·02번의 핵심 함수는 같은 노트북에서 읽고 실행할 수 있습니다.

## 강의 자료

- [초보자를 위한 코드 해설 노트북 3개](hf_colab_gpu/explanations/README.md)
- [Pipeline 추론·셸 시연·스킬 등록 예시](hf_colab_gpu/pipeline-guide.md)
- [실제 T4 셸 시연 결과와 종료 기록](hf_colab_gpu/pipeline-demo-results.md)
- [핵심 코드 네 가지와 조건 변경 실습](hf_colab_gpu/ai-coding-workshop.md)
- [학생 교안](hf_colab_gpu/handson.md)
- [이번 HF·PyTorch GPU 실측 결과](hf_colab_gpu/test-results.md)
- [학생 교안과 검증 결과를 합친 최종 교안](hf_colab_gpu/final-guide.md)
- [새 GitHub Codespaces에서 실제 검증한 결과](hf_colab_gpu/codespaces-verification.md)
- [모델·데이터 출처](THIRD_PARTY.md)
- [양자화 선택 읽을거리](hf_colab_gpu/quantization-reference.md)

기본 데이터는 병·그릇·캔·컵·접시 5개 클래스이며 학습 500장·검증 100장·테스트 200장입니다. 32×32 공개 이미지로 실습 흐름을 배우는 것이므로 실제 작업대 카메라 성능과 구분합니다. 원본 ImageNet 1,000개 라벨 추론과 새 5개 라벨 분류의 정확도를 직접 비교하지 않습니다.

`hf_colab_gpu/validation`은 2026-09-19 기본 과정과 2026-09-21 Pipeline 시연의 보고서·CSV·그림을 담은 참고 폴더입니다. 추가로 실행할 노트북은 없습니다. 각 기록의 날짜와 실행 범위를 확인하고 본인이 실행한 코드의 결과와 구분합니다.

`.devcontainer`, `requirements`, `scripts`, `tests`는 세 노트북의 환경 준비와 검사를 위한 파일입니다. 다운로드한 모델·개인 사진·학습 결과·인증 정보·가상환경은 저장소에 올리지 않습니다.
