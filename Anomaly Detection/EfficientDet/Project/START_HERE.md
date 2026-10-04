# 프로젝트 실행과 결과 확인

프로젝트 위치: `D:\Computer-Vision\Anomaly Detection\EfficientDet\Project`

## 결과 확인

2026-10-04 Colab Tesla T4에서 측정을 완료했다. 검증 60장과 별도 평가 60장을 사용했으며, 검증으로 선택한 설정은 입력 768 / confidence 0.3이다.

- [README](README.md): 실험 개요와 대표 결과
- [실행 노트북](notebooks/vehicle_detection_project.ipynb): 코드, 측정 출력, 단계별 해석과 KPT
- [결과 보고서](results/20261004T094456_281232Z/REPORT.md): 전체 실험 표와 그래프
- [실험 해석과 KPT](docs/PERSONAL_REFLECTION.md): 판단, 한계와 다음 행동
- [제출 상태](docs/SUBMISSION_CHECKLIST.md): 결과 저장 및 공개 확인 상태

## VS Code + Colab에서 다시 실행

1. VS Code에서 Project 폴더와 노트북을 연다.
2. Google Colab·Jupyter 확장을 사용해 **Colab GPU** 커널에 연결한다.
3. 설정과 데이터·정답 박스 출력을 확인하며 위에서 아래로 실행한다. 기본 표본은 검증·평가 각각 60장이다.
4. 새 결과에 맞춰 단계별 해석과 KPT를 갱신한다. 현재 서술은 위 실행 ID의 결과를 기준으로 한다.
5. 마지막 셀에서 생성한 **`vehicle_results_`로 시작하는 ZIP**을 VS Code의 **Colab → Contents → `/content/efficientdet_vehicle_project/results`**에서 다운로드한다.
6. 압축 안의 `results/`를 로컬 Project 폴더에 합치고, 실행 출력이 있는 노트북도 `Ctrl+S`로 저장한다.

로컬 D 드라이브와 Colab 서버의 파일은 별도로 저장된다. 결과 ZIP만 다운로드해도 VS Code에서 열고 있는 노트북이 함께 저장되는 것은 아니다.

## 현재 제출 상태

측정 표·그림·환경 기록·해석·KPT를 정리했다. 공개 저장소 URL과 비로그인 열람 확인 기록은 아직 등록되어 있지 않다. 외부 제출에는 실제 공개 링크 확인이 남아 있다.

오류 대응은 [실행 도움말](docs/TROUBLESHOOTING.md)을 참고한다.
