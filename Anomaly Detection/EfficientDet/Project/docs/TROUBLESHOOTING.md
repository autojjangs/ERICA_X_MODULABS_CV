# VS Code + Colab 실행 도움말

| 증상 | 확인할 사항 |
| --- | --- |
| `CUDA GPU가 없습니다` | 현재 커널이 로컬 Python인지 Colab인지 확인하고 GPU 런타임으로 변경. 계정의 GPU 할당 여부 확인 |
| `No module named torch` | Colab 커널을 선택. 이 프로젝트는 Colab에 있는 torch/torchvision 조합을 사용하며 무작정 교체하지 않음 |
| `torchvision::nms` 오류 | torch와 torchvision 조합 문제. 새 Colab GPU 런타임에서 시작해 기본 설치 조합 유지 |
| Kaggle 다운로드 인증·접속 오류 | 직접 Kaggle 로그인 또는 받은 데이터를 Colab에 업로드하고 `dataset_root`에 서버 경로 지정 |
| COCO 주석 파일을 못 찾음 | 파일 목록을 보고 `annotation_root`를 한 COCO 버전 폴더로 지정 |
| `No target boxes` | `categories`의 실제 이름을 확인하고 `target_category_names` 수정. 다른 차종을 승용차로 임의 병합하지 않음 |
| 테스트 분할이 비어 있음 | 프레임 번호와 수량 확인. 측정 전에 표본 수·분할 간격을 조정하고 변경 이유 기록 |
| `Source snapshot differs` | 새 원격 ROOT 경로를 사용하거나 의도적으로 업로드한 코드에 `USE_UPLOADED_PROJECT=True` 적용 |
| VS Code에서 결과 파일을 찾지 못함 | 로컬 탐색기가 아니라 Colab 확장의 Contents에서 `/content/efficientdet_vehicle_project/results/` 확인 |
| `files.download()`가 동작하지 않음 | 이 노트북은 해당 웹 전용 기능을 사용하지 않음. Colab Contents의 Download 기능 사용 |
| 실행 시간이 길어짐 | GPU 확인 후 새 실행에서 검증/평가를 각각 30장으로 축소. 기존 결과와 표본 수를 섞지 않음 |

실행 중 오류가 나면 오류 셀과 마지막 출력, `config.json`의 변경 사항을 함께 기록합니다. 오류가 난 실행을 완료한 실험으로 표시하지 않습니다.
