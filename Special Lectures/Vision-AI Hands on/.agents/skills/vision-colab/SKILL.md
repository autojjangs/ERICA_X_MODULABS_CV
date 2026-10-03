---
name: vision-colab
description: vision_ai_notbook에서 Hugging Face CLI로 모델을 준비하고 Transformers·PyTorch 노트북을 Colab CLI GPU 세션에서 실행해 결과를 회수한다. 문서·환경 수정만 요청하면 원격 자원을 만들지 않는다.
---

# Vision AI Colab 실습

`.vision-lab-root`로 프로젝트를 찾는다. 기본 과정의 명령·경로·설치 파일은 [Hugging Face CLI + Colab GPU 안내](../../../hf_colab_gpu/README.md)를 기준으로 삼는다. 모델·학습 코드는 노트북에 보이는 셀을 그대로 사용하고, 별도 통합 실행 함수나 프로젝트 패키지로 감싸지 않는다.

Hub 모델의 `pipeline()` 추론을 `.sh`로 시연하는 요청은 [Pipeline 안내](../../../hf_colab_gpu/pipeline-guide.md)와 [vision-pipeline-inference](../vision-pipeline-inference/SKILL.md)를 사용한다. 이 경로는 별도 추론 시연이며 아래 노트북 학습 흐름을 대체하지 않는다.

1. 프로젝트의 `.venv/bin/python scripts/doctor.py`로 Python과 라이브러리 경로를 확인한다. 가상환경이 없으면 `bash scripts/setup.sh`로 기본 공개 라이브러리를 설치한다. `hf_colab_gpu/notebooks/00_hf_download_and_data.ipynb`는 Codespaces CPU에서 실행한다. 이 노트북의 `hf download`로 모델 원본을 받고, 다운로드 기록과 데이터를 준비한 뒤 `doctor.py --data`로 검사한다.
2. 계획 요청에는 실행 위치·설정·필요 파일·공식 CLI 명령을 보여 준다. 실제 원격 실행 요청이 있을 때만 세션을 만든다. 01·02에는 완성된 핵심 함수 네 개가 들어 있다. GPU 실행 전에 `python scripts/check_student_cells.py`로 문법과 함수 정의를 확인한다. 사용자가 코드를 수정했다면 입력 노트북을 저장한 뒤 다시 검사한다. GPU 실행 순서는 `hf_colab_gpu/notebooks/01_gpu_inference.ipynb` → `02_gpu_finetuning.ipynb`다. Codespaces에서 실행한 Python을 Colab GPU 실행으로 취급하지 않는다.
3. 먼저 터미널에서 `colab --auth oauth2 sessions`로 인증과 기존 세션을 확인한다. CLI 0.6.0에는 `login` 명령이 없다. 필요한 인증 코드는 사용자가 터미널에 입력하며, 인증을 끝내기 전에 표준 입력으로 실행 코드를 전달하지 않는다. 기존 ADC 사용자는 준비된 인증을 유지하고 모든 관련 명령에 `--auth adc`를 적용한다.
4. 기본 안내의 세션 생성 → requirements 두 파일 원격 업로드 → `colab exec`에서 설치 → 커널 재시작 → GPU 확인 → 입력 폴더 준비 순서를 사용한다. `hf_colab_gpu/requirements-torch.txt`와 `hf_colab_gpu/requirements-gpu.txt`의 설치는 README의 긴 대기시간을 둔 `colab exec` 명령을 따른다. CLI 0.6.0의 `colab install`은 큰 PyTorch 다운로드 도중 내부 제한 시간에 걸릴 수 있다. 새 세션 이름을 기록하고 기존 세션을 덮어쓰지 않는다. `torch.cuda.is_available()`이 거짓이면 멈추고 CPU로 대체하지 않는다. `.vision-lab-root`, 모델 원본 3개 파일(`config.json`, `preprocessor_config.json`, `pytorch_model.bin`)과 `download_manifest.json`, 데이터 4개 파일(`manifest.json`, `train.npz`, `validation.npz`, `test.npz`)을 각각 안내된 경로에 업로드한다.
5. 모델·데이터 체크섬과 고정 모델 revision 검사를 유지한다. 설정 변경이 요청되면 노트북의 `BATCH_SIZE`, `HEAD_EPOCHS`, `FINETUNE_EPOCHS`, `HEAD_LR`, `FINETUNE_LR`을 직접 바꾼다. 완료된 `report.json`이 있으면 이전 결과와 출력 노트북을 보존한 뒤 새 실험을 시작한다. 최적 모델은 검증 정확도, 동률이면 검증 손실로 선택하며 최종 평가 데이터로 선택하지 않는다.
6. `colab exec -s 세션이름 -f hf_colab_gpu/notebooks/01_gpu_inference.ipynb --timeout 1800`으로 추론을 실행한다. `*_output.ipynb`의 모든 셀에 오류가 없는지 확인한 뒤 02번을 실행한다. CLI 0.6.0은 셀 오류 후에도 진행할 수 있으므로 종료코드나 마지막 완료 문구만으로 성공을 판단하지 않는다. `--timeout`은 셀의 대기시간이며 자동 세션 종료 설정이 아니다.
7. 기본 결과는 `hf_colab_gpu/results/gpu`로 파일별 회수한다. 추론 보고서·그림과 학습 보고서의 `artifacts`, `finetuned_model`을 포함하고 `artifact_sha256`도 확인한다. 학습 보고서는 `status='completed'`, `platform='gpu'`, 실제 GPU 이름, `verification.frozen_parameters_unchanged`, `verification.last_block_changed`, 저장 모델 재로딩 검증과 오류 없는 출력 노트북을 함께 확인한다. 과거 검증 기록을 현재 코드의 실행 결과로 사용하지 않는다.
8. 실행·다운로드 오류가 나면 해당 세션의 상태와 남은 파일부터 확인한다. 새 자원을 반복 할당하지 않는다. 필요한 파일 회수를 시도하고, 복구를 중단할 때에도 이번 작업에서 만든 세션은 `colab stop -s 이름`으로 종료한다. `colab sessions`로 종료를 확인하고 다른 작업의 세션은 건드리지 않는다. 회수하지 못한 파일과 미완료 항목은 그대로 보고한다.
9. 보고서·체크포인트·CSV·출력 노트북의 위치와 세션 종료 확인 결과를 전달한다. Codespaces로 회수한 파일과 학생 PC에 내려받은 파일을 구분한다. 설명·환경 수정만으로 원격 자원을 만들거나 CU를 구매하지 않는다.
