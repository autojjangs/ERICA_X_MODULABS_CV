# 정상 타일에 균열 데이터 만들기

정상 타일의 작은 영역에 Stable Diffusion Inpainting으로 균열을 생성한다. 같은 원본, 마스크, 시드를 사용하면서 생성 강도만 0.65와 0.95로 바꿔 비교하는 컴퓨터비전 실습이다.

기본 조건은 정상 이미지 2장 × 시드 2개 × 강도 2개, 총 8장이다. 모델을 새로 학습하지 않아 작은 프로젝트로 진행할 수 있다.

## 실행

1. `tile_crack_inpainting.ipynb`를 VS Code에서 연다.
2. Jupyter 확장과 [Google Colab 확장](https://marketplace.visualstudio.com/items?itemName=google.colab)을 설치한다.
3. **Select Kernel → Colab → Auto Connect**에서 GPU 서버에 연결한다.
4. 셀을 위에서 아래로 실행한다. GPU 확인 셀에서 장치명을 확인한다.
5. 결과를 확인하고 출력이 포함되도록 ipynb를 저장한다.

노트북 첫 설치 셀만 실행하면 되며 `requirements.txt`를 별도로 설치할 필요는 없다. 이미 설치한 패키지를 불러온 상태였다면 설치 뒤 커널을 재시작하고 처음부터 실행한다. 기존 Colab의 GPU용 PyTorch와 CUDA를 사용한다.

처음 모델을 받을 때 수 GB의 다운로드가 필요하다. 처음 동작만 확인하려면 설정을 `NUM_NORMALS = 1`, `SEEDS = [17]`로 바꾼다. 이때는 2장을 생성한다.

## 무엇을 비교하나

- 마스크 안에서 픽셀이 얼마나 바뀌었는지
- 모델 원출력에서 마스크 밖 표면이 얼마나 바뀌었는지
- 균열이 보이는지, 자연스러운지, 합성 경계가 보이는지에 대한 직접 평가

최종 이미지의 마스크 밖은 원본으로 복원한다. 이 부분의 변화량 0은 후처리 확인값이다. 생성 모델의 품질 점수는 아니다. 마스크는 수정을 요청한 영역이며 실제 균열의 정확한 분할 정답은 아니다.

## 만들어지는 결과

노트북 실행 위치 아래 `tile_crack_runs/실행시각/`에 저장된다. Colab에서는 보통 `/content/tile_crack_runs/`이다.

| 파일 또는 폴더 | 내용 |
| --- | --- |
| `inputs/`, `masks/` | 원본, 크기를 맞춘 이미지, 생성 영역, 합성용 alpha |
| `raw/`, `images/` | 모델 원출력, 마스크 밖을 복원한 최종 이미지 |
| `figures/` | 원본과 마스크 미리보기, 조건별 비교, 변화량 그림 |
| `metrics.csv`, `summary.csv` | 샘플별 측정값과 조건별 요약 |
| `result_interpretation.md` | 실제 측정값에 따른 조건별 해석 |
| `my_review.csv` | 직접 입력한 가시성, 자연스러움, 경계 흔적 점수 |
| `config.json`, `scheduler_config.json` | 생성 설정 |
| `input_manifest.json`, `generation_manifest.json` | 입력 출처, 파일 해시, 생성 기록 |
| `environment.json`, `environment_freeze.txt` | 실행한 GPU와 패키지 환경 |
| `DATA_LICENSE_AND_SOURCES.txt` | 이미지 출처와 사용 조건 |

마지막 ZIP 셀을 실행하면 이 결과를 압축한다. **ipynb는 ZIP에 자동 포함되지 않으므로 따로 저장한다.** VS Code에서 ZIP 링크가 열리지 않으면 원격 파일 탐색기로 내려받거나 노트북의 선택 셀로 Google Drive에 복사한다.

## GitHub

GitHub에는 이 노트북과 README를 올리고, 실행 결과 중 비교 그림·측정표·직접 평가표·설정·환경·출처 파일을 `results/` 폴더로 옮겨 함께 올리면 된다. 원본 전체 데이터셋, 모델 가중치, 결과 ZIP은 제외한다. ipynb는 실행 출력이 보이는 상태로 저장한다.

## 데이터와 모델 출처

- [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), Paul Bergmann 외, CVPR 2019. 정상 tile 학습 이미지 2장을 사용한다. 원본과 변형 이미지는 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 조건에 따라 비상업 교육 목적으로 이용한다. 이미지 변형은 512px 크기 조정과 국소 균열 생성·합성이다.
- [Voxel51 공개 미러](https://huggingface.co/datasets/Voxel51/mvtec-ad). 노트북에서 데이터 리비전과 원본 해시를 고정한다.
- [Stable Diffusion Inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting), CreativeML OpenRAIL-M. 모델 리비전도 고정한다.
- [Diffusers inpainting API](https://huggingface.co/docs/diffusers/v0.35.1/en/api/pipelines/stable_diffusion/inpaint)와 [메모리 절약 안내](https://huggingface.co/docs/diffusers/v0.35.1/en/optimization/memory).

수업에서 제공한 PDF와 ipynb는 개념과 주제 선택에 참고했다. 수업의 외부 연구 코드를 복제하지 않고 공식 inpainting API로 구성했다. 검사 모델을 학습하거나 실제 결함 테스트셋으로 평가하지 않았으므로 검사 정확도가 좋아졌다고 주장하지 않는다.

## 실행 문제 해결

GPU가 없다는 오류가 나오면 커널과 서버의 GPU 할당을 확인한다. 메모리가 부족하면 다른 모델을 실행하지 않은 새 커널에서 기본값으로 다시 실행한다. `librosa` 또는 `np.complex` 관련 import 오류가 나오면 `librosa>=0.10.2`로 갱신하고 커널을 다시 시작한다.
