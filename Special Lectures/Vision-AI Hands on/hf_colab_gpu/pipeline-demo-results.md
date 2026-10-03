# Pipeline + Colab 셸 시연 · 실제 실행 기록

2026-09-21에 `run_pipeline_colab.sh`로 **새 Colab Tesla T4 세션 하나**를 만들어 Hugging Face Hub 모델을 추론했습니다. 결과 JSON을 회수·검사했고 해당 세션이 목록에서 사라진 것까지 확인했습니다.

## 실행한 구성

| 항목 | 실제 값 |
|---|---|
| 모델 | `facebook/deit-tiny-patch16-224` |
| 모델 revision | `b3428f18dcc7b543470d07f14b4a4157815d1880` |
| 추론 API | `transformers.pipeline("image-classification")` |
| 실행 장치 | `Tesla T4` / `cuda` |
| PyTorch | `2.8.0+cu126` |
| Transformers | `4.57.6` |
| Hugging Face Hub | `0.36.0` |
| 입력 | CIFAR-100 테스트 이미지 `cifar100/test/04570`, 정답 `cup`, 32×32 |
| 실행 ID | `76443fdd48a34bf099fcfadfc9ea2f0d` |
| 종료 | `termination_status="confirmed"`, `result_verified=true`, `success=true` |

입력은 00번과 같은 공개 데이터에서 고른 사진입니다. 개인 사진이나 인증 파일을 업로드하지 않았습니다. [입력과 출처 기록](validation/pipeline-demo/public-input.json)에 데이터 파일과 이미지의 SHA-256을 남겼습니다.

<img src="validation/pipeline-demo/input.png" alt="실제 시연에 사용한 CIFAR-100 cup 이미지" width="224">

## 실제 Top-5

| 순위 | 원본 모델의 예측 라벨 | softmax 점수 |
|---|---|---|
| 1 | face powder | 5.0141% |
| 2 | eggnog | 4.0992% |
| 3 | frying pan, frypan, skillet | 1.8729% |
| 4 | waffle iron | 1.7263% |
| 5 | potter's wheel | 1.3867% |

원본 모델은 ImageNet 1,000개 라벨을 사용합니다. 실습 정답이 `cup`이어도 같은 이름을 출력하거나 높은 점수를 준다고 보장하지 않습니다. 이 표는 이미지 한 장의 추론 결과이며 데이터셋 정확도가 아닙니다. 작업대 물체 5종에 맞추는 학습은 02번에서 진행합니다.

## 같은 사진으로 시연하기

프로젝트 최상위의 Codespaces 터미널에서 환경을 활성화하고 본인 Google 인증을 마친 뒤 실행합니다.

```bash
source .venv/bin/activate
colab --auth oauth2 sessions
bash hf_colab_gpu/run_pipeline_colab.sh \
  --image hf_colab_gpu/validation/pipeline-demo/input.png
```

이번 검증에서는 이미 준비된 ADC 인증을 사용해 같은 셸 파일에 `--auth adc`와 입력 이미지 경로를 전달했습니다. 스크립트가 requirements 설치, 커널 재시작, Hub 모델 다운로드, GPU 추론, JSON 회수와 검증, 세션 종료를 수행했습니다. 기본 실행도 항상 새 T4를 사용하므로 실제 재실행에는 Colab CU가 필요합니다.

실제 로그에서 표시된 완료 단계는 다음과 같습니다.

```text
1/6 인증과 세션 이름 확인
2/6 새 T4 세션 생성
3/6 코드와 이미지 전송
4/6 GPU 라이브러리 설치
5/6 Hub 모델 추론
6/6 결과 회수·검사
시연 완료: T4 추론 결과와 세션 종료를 모두 확인했습니다.
```

위 단계 문장은 원시 로그를 읽기 쉽게 줄인 것입니다. 검증 근거는 [원본 결과 JSON](validation/pipeline-demo/result.json)과 [종료 기록](validation/pipeline-demo/cleanup.json)입니다. 원시 CLI 로그와 인증 상태 파일은 저장소에 포함하지 않습니다.

## 검증 범위

- 실제 `.sh` 전체 경로와 Hub `pipeline()` 추론을 T4에서 실행했습니다.
- 이미지·코드·requirements의 SHA-256, 모델 revision, 설치 버전, 결과 형태와 종료 상태를 검사했습니다.
- 로컬 자동 테스트 80개를 통과했습니다. 여기에는 원격 오류가 종료 코드 0으로 보고되는 경우, 잘못된 결과, 생성 부분 실패, 종료 실패 등을 흉내 낸 셸 테스트도 포함합니다.
- 이번 변경에서 새로운 Codespaces를 만들거나 02번 전체 파인튜닝을 다시 수행하지는 않았습니다. 00의 기존 실행 코드는 유지했고 환경 안내만 앞에 추가했습니다.
- 원래 T4 학습 검증은 [2026-09-19 기본 과정 기록](test-results.md)과 구분합니다.
