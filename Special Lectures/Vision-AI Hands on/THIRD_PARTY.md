# 모델과 데이터 출처

이 저장소에는 학습 코드, 강의 문서와 과거 실행의 검증 기록을 포함한다. 사전학습 가중치와 이미지 데이터는 저장소에 포함하지 않고 00번 노트북에서 내려받는다. 다운로드할 파일은 아래 버전과 체크섬으로 식별한다.

| 항목 | 출처와 고정 버전 | 배포·변환 내용 |
|---|---|---|
| 사전학습 모델 | [facebook/deit-tiny-patch16-224](https://huggingface.co/facebook/deit-tiny-patch16-224), revision `b3428f18dcc7b543470d07f14b4a4157815d1880` | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0). 학생이 HF CLI로 원본 `.bin`과 설정을 직접 다운로드. 다운로드 파일의 해시는 00번 노트북에서 검증하며 `hf_colab_gpu/models/deit-tiny/download_manifest.json`에 기록 |
| 모델 구조 | [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877), Hugo Touvron 외 | Transformers의 ViT 분류 모델로 로딩 |
| 원본 이미지 데이터 | [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), Alex Krizhevsky, Vinod Nair, Geoffrey Hinton | 100개 클래스 중 식품 용기 5개 클래스를 사용. 원본은 32×32 이미지이며 실제 작업대 카메라 데이터가 아님 |
| 다운로드 미러 | [uoft-cs/cifar100](https://huggingface.co/datasets/uoft-cs/cifar100), revision `aadb3af77e9048adbea6b47c21a81e47dd092ae5` | train/test Parquet의 게시된 SHA-256을 검증하고 필요한 이미지 선택. 원본 데이터 저작자·출처를 유지하며 별도 재라이선스를 주장하지 않음 |
| HF CLI | [공식 0.36.0 CLI 문서](https://github.com/huggingface/huggingface_hub/blob/v0.36.0/docs/source/en/guides/cli.md) | `huggingface_hub[cli]==0.36.0`. 원본 파일명·revision·저장 경로를 명시 |
| Transformers | [공식 4.57.6 소스](https://github.com/huggingface/transformers/tree/v4.57.6) | 4.57.6. 기본 과정의 AutoImageProcessor·AutoModelForImageClassification·save_pretrained 사용 |
| PyTorch·Torchvision | [공식 버전 조합](https://pytorch.org/get-started/previous-versions/) | 2.8.0·0.23.0, CUDA 12.6 배포처. Colab GPU에 설치 |
| Colab CLI | [공식 저장소](https://github.com/googlecolab/google-colab-cli/tree/v0.6.0) | 0.6.0. 학생이 공식 명령을 단계별로 사용하며 Google의 공식 수업 패키지는 아님 |

실습은 다운로드한 원본 가중치를 그대로 읽으며, 추가 학습 결과를 `save_pretrained`로 별도 폴더에 저장한다. 전처리는 일반적인 ImageNet 정규화 값을 임의로 적용하지 않고 이 체크포인트에 포함된 `preprocessor_config.json`의 평균·표준편차 0.5를 따른다. 모델과 데이터를 별도로 배포할 때에도 원저작자·라이선스·출처를 함께 확인한다.

CIFAR-100 사용 시 참고 문헌: Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009. [기술 보고서](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

학생의 실제 작업대 이미지는 제공자가 사용 권한을 확인해 별도 폴더로 준비한다. 원본 개인 이미지는 기본 저장소나 배포 ZIP에 자동 포함하지 않는다.
