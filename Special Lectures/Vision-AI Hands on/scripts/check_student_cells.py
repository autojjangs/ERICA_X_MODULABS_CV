"""Check required function cells without importing Torch or executing any cell."""
import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    '01_gpu_inference.ipynb': {'01-infer': 'infer_probabilities', '01-topk': 'topk_predictions'},
    '02_gpu_finetuning.ipynb': {'02-step': 'train_one_batch', '02-scope': 'select_finetune_parameters'},
}


def check_notebook(path):
    path = Path(path)
    try:
        notebook = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        return [f'{path.name}: 파일을 읽을 수 없습니다: {error}']
    expected = EXPECTED.get(path.name)
    if expected is None:
        return [f'{path.name}: 기본 과정 01 또는 02 노트북을 지정하세요.']
    issues, found = [], {}
    for number, cell in enumerate(notebook.get('cells', []), 1):
        if cell.get('cell_type') != 'code':
            continue
        source = cell.get('source', '')
        source = source if isinstance(source, str) else ''.join(source)
        metadata = cell.get('metadata', {})
        is_task = bool({'provided-function', 'student-task'} & set(metadata.get('tags', [])))
        label = f'{path.name} · {number}번째 셀'
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            issues.append(f'{label}: 문법 오류: {error.msg} (줄 {error.lineno})')
            continue
        if not is_task:
            continue
        identifier = metadata.get('exercise_id')
        function = expected.get(identifier)
        if function is None or identifier in found or metadata.get('expected_function') != function:
            issues.append(f'{label}: 함수 셀의 이름·메타데이터를 확인하세요.')
            continue
        found[identifier] = True
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function]
        if not definitions:
            issues.append(f'{label}: {function} 함수가 아직 없습니다. 함수 정의를 확인하고 저장하세요.')
        elif not any(not isinstance(n, (ast.Pass, ast.Expr)) or
                     (isinstance(n, ast.Expr) and not isinstance(n.value, ast.Constant))
                     for n in definitions[0].body):
            issues.append(f'{label}: {function} 함수가 pass·설명만 있습니다. 함수 본문을 확인하세요.')
    missing = set(expected) - set(found)
    if missing:
        issues.append(f'{path.name}: 함수 셀 누락 또는 문법 오류: {", ".join(sorted(missing))}')
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebooks', nargs='*', type=Path)
    args = parser.parse_args()
    paths = args.notebooks or [ROOT / 'hf_colab_gpu/notebooks' / name for name in EXPECTED]
    issues = [issue for path in paths for issue in check_notebook(path)]
    if issues:
        print('아직 GPU 실행 준비가 끝나지 않았습니다.')
        for issue in issues:
            print('- ' + issue)
        return 1
    print(f'사전 검사 통과: 노트북 {len(paths)}개. 문법과 함수 정의를 확인했습니다.')
    print('계산 결과나 GPU 동작을 보장하는 검사는 아닙니다. 실행 후 각 확인 셀과 출력의 오류를 확인하세요.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
