"""Combine the basic course handout with its own measured verification record."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
intro = "# HF CLI와 Colab GPU · 최종 강의 교안\n\n`notebooks`의 00·01·02번으로 진행하는 Transformers·PyTorch 실습입니다. 01·02번의 핵심 함수 네 개는 완성된 코드로 제공합니다. 뒤의 실측 결과는 2026-09-19 실행 기록을 당시 설명과 함께 보존한 것입니다. 아래 기록의 학생용 빈칸 관련 설명은 당시 버전에 해당하며, 이번 완성 코드 변경 후 새로 GPU를 실행한 기록은 아닙니다.\n\n"
parts = [(HERE / name).read_text(encoding="utf-8") for name in ("handson.md", "test-results.md")]
(HERE / "final-guide.md").write_text(intro + "\n\n---\n\n".join(parts), encoding="utf-8")
