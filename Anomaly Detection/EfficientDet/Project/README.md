# EfficientDet 차량 탐지: 정확도·속도·영상 열화 비교

같은 EfficientDet-D0 가중치에서 입력 해상도와 검출 임계값을 바꾸고, 어두움·흐림이 차량 누락에 미치는 영향을 비교하는 컴퓨터비전 프로젝트입니다. 공장 출입구·물류 구역의 차량 모니터링을 응용 시나리오로 삼으며, 실험은 공개 차량 이미지에서 수행합니다.

**실험 완료: 2026-10-04, Colab Tesla T4.** 검증 60장으로 설정을 선택하고 별도 60장에서 최종 평가했습니다. 검증 F1이 가장 높은 **입력 768 / confidence 0.3**을 선택했습니다.

| 원본 최종 평가 | 기본 512/0.3 | 선택 768/0.3 |
| --- | ---: | ---: |
| Precision | 0.7350 | 0.7159 |
| Recall | 0.3654 | 0.5354 |
| F1 | 0.4882 | 0.6126 |
| AP50 | 0.4641 | 0.5937 |
| 작은 차량 Recall | 0.3054 | 0.4930 |
| 프레임별 차량 수 MAE | 7.45대 | 6.20대 |
| 중앙 처리 시간 | 54.33ms | 62.91ms |

F1은 12.45%p 높아졌고 처리 시간은 약 15.8% 늘었습니다. 선택 설정의 어두움·흐림 F1은 각각 0.6156, 0.5820입니다. 어두움에서는 오탐 감소로 F1이 소폭 올랐지만 Recall은 낮아졌습니다. 상세 수치와 한계는 [결과 보고서](results/20261004T094456_281232Z/REPORT.md), 판단과 다음 실험 계획은 [실험 해석과 KPT](docs/PERSONAL_REFLECTION.md)에 정리했습니다.

## VS Code + Colab에서 시작

1. 이 프로젝트의 파일을 `D:\Computer-Vision\Anomaly Detection\EfficientDet\Project`에 두고, VS Code에서 해당 폴더를 엽니다.
2. Google의 **Colab** 확장과 Jupyter 확장을 준비합니다. `notebooks/vehicle_detection_project.ipynb`를 엽니다.
3. 커널 선택에서 **Colab**을 선택하고 GPU 런타임에 연결합니다. 첫 실행 셀이 GPU와 Python 환경을 확인합니다. 계정 로그인은 직접 수행합니다.
4. 노트북을 위에서 아래로 실행합니다. 코드·설정의 작은 스냅샷이 노트북에 포함되어 있어 별도 GitHub 저장소 없이 Colab에서 시작할 수 있습니다.
5. 실험 후 VS Code의 **Colab → Contents**에서 `/content/efficientdet_vehicle_project/results/vehicle_results_....zip`을 찾아 **Download...**로 로컬에 저장합니다. 압축 안의 `results/`를 이 프로젝트에 합칩니다.
6. VS Code에서 실행된 노트북을 저장합니다(`Ctrl+S`). ZIP에는 그래프·표·보고서·실행 기록이 들어가며, 현재 편집기의 노트북 출력은 별도로 저장해야 합니다.

로컬 D 드라이브 경로와 Colab의 `/content/...`는 서로 다른 컴퓨터의 경로입니다. Colab 코드에 Windows 경로를 데이터 경로로 넣지 않습니다. 확장 버전에 따라 메뉴 표시가 다를 수 있으며, 공식 확장은 파일 업로드·Contents 파일 다운로드를 제공합니다. [공식 확장](https://marketplace.visualstudio.com/items?itemName=Google.colab), [공식 사용 안내](https://github.com/googlecolab/colab-vscode/wiki/User-Guide).

## 실험 설계

| 항목 | 설정 |
| --- | --- |
| 모델 | COCO 사전학습 EfficientDet-D0, FP32, batch 1, 추가 학습 없음 |
| 탐지 대상 | `Car` 주석과 COCO의 `car` 슬롯. 대소문자는 무시하되 `cars`·`Pickup`은 자동 병합하지 않음 |
| 검증 | 최대 60장, 입력 512/768 × confidence 0.1/0.3/0.5 = 6조건 |
| 선택 기준 | 원본 검증 F1 최대, 정확히 동률이면 측정된 중앙 처리 시간이 짧은 조건 |
| 최종 평가 | 별도 최대 60장, 기본 설정(512/0.3)과 선택 설정을 고정 |
| 영상 조건 | 원본, 밝기 0.6배, Gaussian blur 5×5·sigma 1.0 |
| 박스 평가 | 일대일 IoU ≥ 0.5 대응, Precision·Recall·F1, AP50 |
| AP 검증 | score floor 0.001·최대 100개 예측의 101점 AP50을 pycocotools와 대조 |
| 보조 지표 | 프레임별 승용차 수 MAE, 상대 면적 1% 이하 차량의 Recall |
| 속도 | 각 실제 운영 임계값에서 준비 실행 5회 후 최대 10장 × 3회 반복 |

AP50은 임계값별 F1과 용도가 다릅니다. AP50용 예측은 낮은 공통 score floor로 수집하므로 같은 해상도에서는 임계값 행마다 AP50이 같습니다. 임계값 효과는 Precision·Recall·F1에서 확인합니다. AP@[0.5:0.95]는 이 프로젝트의 지표가 아닙니다.

속도는 메모리에 있는 이미지의 전처리·전송·모델·후처리·CPU 결과 변환을 포함하며 CUDA 동기화를 적용합니다. 파일 읽기, 합성 열화 생성, 시각화, 영상 디코딩은 제외합니다. `memory_pipeline_fps`를 실제 CCTV 시스템의 FPS로 해석하지 않습니다. 조건 순서는 시드로 섞으며 원시 반복 측정값도 저장합니다.

## 데이터 준비와 분할

[Vehicle Detection Image Dataset](https://www.kaggle.com/datasets/pkdarabi/vehicle-detection-image-dataset)을 KaggleHub로 다운로드합니다. `No_Apply_Grayscale` 안의 한 COCO 버전만 사용합니다. 다운로드된 구조와 클래스 이름이 다르면 임의로 대체하지 않고 오류와 수정 지점을 안내합니다.

- 노트북 설정 셀의 `dataset_root`로 이미 다운로드한 **Colab 서버 쪽 경로**를 지정할 수 있습니다.
- `annotation_root`를 설정하면 해당 COCO 버전의 `train/valid/test`를 검색합니다.
- `data_audit.json`에서 실제 클래스별 주석 수와 `Car` 대응을 확인합니다. 주석이 없거나 손상되면 평가를 중단합니다.
- 원본 파일명과 픽셀 해시로 동일·파생 이미지를 제거합니다. 모든 파일에서 프레임 번호를 읽을 수 있으면 앞 시간 구간을 검증, 뒤 구간을 최종 평가에 사용하고 경계에 30프레임 간격을 둡니다.
- 프레임 번호를 알 수 없으면 원본 이미지 그룹을 나눕니다. 이때 장면 유사성에 의한 잔여 누수 가능성을 기록합니다.
- 추가 학습이 없으므로 원본 데이터의 train/valid/test를 합친 뒤 평가용 두 분할을 새로 만듭니다. 제공 데이터셋의 공식 점수와 직접 비교하지 않습니다. 분할 목록·이미지 해시·주석 해시를 저장합니다.
- 숫자 프레임 정보만으로 카메라나 장면이 독립임을 보장할 수 없습니다. 같은 카메라의 다른 시간 구간 실험일 수 있습니다.

다운로드에 계정 인증이 요구되면 직접 Kaggle에 로그인하거나 이미 받은 파일을 Colab에 업로드해 `dataset_root`를 지정합니다. 비밀번호·API 키를 노트북이나 저장소에 적지 않습니다.

## 단계별 산출물

```text
Project/
├── README.md
├── notebooks/vehicle_detection_project.ipynb
├── configs/experiment.json
├── src/vehicle_project/
│   ├── data.py           # COCO 검증·중복 제거·분할
│   ├── inference.py      # D0 추론·전처리·속도 측정
│   ├── metrics.py        # 일대일 대응·AP50·개수 오차
│   ├── experiment.py     # 검증 → 설정 고정 → 최종 평가
│   └── report.py         # 결과 표·그림·관찰 보고서
├── tests/                # 합성 fixture 기반 평가·분할 검증
├── docs/                 # 실험 회고·제출 점검·실행 안내
├── tools/build_notebook.py
└── results/20261004T094456_281232Z/
    ├── REPORT.md
    ├── validation.csv / test.csv
    ├── figures/
    ├── config.json / selection.json
    ├── split_manifest.json / data_audit.json
    ├── model_provenance.json / environment.json
    ├── timings/          # 반복 속도 측정값
    └── run_status.json
```

실행마다 다른 결과 폴더를 사용합니다. 중단된 실행은 부분 결과만 남으며 완료 결과로 표시하지 않습니다. 모델 코드의 실제 commit, 가중치 SHA-256, 데이터의 해석된 버전을 저장합니다. 첫 실행은 원본 `master`를 가져오며, 재현할 때는 완료 실행의 `config.json`에 저장된 commit과 데이터 버전을 사용합니다. 저장된 환경 버전도 함께 맞춰야 합니다.

로컬 `src/`를 수정했으면 로컬 Python에서 `python tools/build_notebook.py`로 노트북에 포함된 코드 스냅샷을 갱신합니다. 이 명령은 노트북의 포함 소스만 갱신하며 마크다운과 측정 출력은 보존합니다. 소스 준비 셀의 이전 실행 출력은 갱신된 코드와 다르므로 초기화합니다. 이미 Colab에 올린 수정 코드를 쓰려면 노트북의 `USE_UPLOADED_PROJECT=True`를 선택합니다.

## 재현·검증

Colab 셀에서 오프라인 단위 검증도 실행합니다. 로컬 의존성을 갖춘 환경에서는 프로젝트 루트에서 다음을 실행할 수 있습니다.

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
python -m vehicle_project.experiment --root . --config configs/experiment.json
```

로컬 GPU 없이 실험하려면 `require_cuda=false`를 명시해야 합니다. CPU 속도와 GPU 속도를 같은 실험의 전후 성능으로 비교하지 않습니다. 개발용 합성 fixture의 수치는 실제 모델 결과에 포함하지 않습니다.

## 참고 및 한계

- [EfficientDet 원 논문](https://arxiv.org/abs/1911.09070), [사용 구현](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch), [COCO 평가 구현](https://github.com/cocodataset/cocoapi).
- 직접 학습한 모델, 차체 결함 검사, 실제 야간·우천 주행 평가, 고유 차량 추적·누적 통과 대수 시스템은 이 결과의 범위가 아닙니다.
- 데이터와 원본 모델 코드는 각각의 출처 및 이용 조건을 따릅니다. 원본 모델은 실행 시 별도 내려받으며 원본 LICENSE를 보존합니다. 전체 데이터·가중치는 저장소에 포함하지 않습니다.
