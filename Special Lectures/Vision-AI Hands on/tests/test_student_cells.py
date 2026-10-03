"""Validate completed function cells and fail-closed execution."""
import ast
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

from scripts.check_student_cells import EXPECTED, check_notebook


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "hf_colab_gpu/notebooks"


def read_notebook(name):
    return nbformat.read(NOTEBOOKS / name, as_version=4)


def minimal_notebook(tmp_path, name):
    # These deliberately unimplemented definitions test syntax, never give answers.
    cells = [nbformat.v4.new_code_cell(
        f"def {function}(*args, **kwargs):\n    raise NotImplementedError",
        metadata={"tags": ["provided-function"], "exercise_id": identifier,
                  "expected_function": function},
    ) for identifier, function in EXPECTED[name].items()]
    path = tmp_path / name
    nbformat.write(nbformat.v4.new_notebook(cells=cells), path)
    return path


@pytest.mark.parametrize("name", EXPECTED)
def test_notebook_has_two_implemented_functions_and_valid_code(name):
    notebook = read_notebook(name)
    nbformat.validate(notebook)
    tasks = [cell for cell in notebook.cells
             if "provided-function" in cell.metadata.get("tags", [])]
    assert len(tasks) == 2
    assert {cell.metadata.exercise_id: cell.metadata.expected_function
            for cell in tasks} == EXPECTED[name]
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert cell.outputs == [] and cell.execution_count is None
    assert not any("student-task" in cell.metadata.get("tags", []) for cell in notebook.cells)
    assert all(cell.source.strip() for cell in notebook.cells if cell.cell_type == "code")
    for cell in tasks:
        tree = ast.parse(cell.source)
        assert any(isinstance(node, ast.FunctionDef) and node.name == cell.metadata.expected_function
                   for node in tree.body)
    assert check_notebook(NOTEBOOKS / name) == []


@pytest.mark.parametrize("name", EXPECTED)
def test_gpu_setup_rejects_missing_cuda_instead_of_using_cpu(name):
    source = next(cell.source for cell in read_notebook(name).cells
                  if cell.cell_type == "code" and "if not torch.cuda.is_available():" in cell.source)
    tree = ast.parse(source)
    guard = next(node for node in tree.body if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "not torch.cuda.is_available()")
    device = next(node for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "device"
                          for target in node.targets))
    setup = compile(ast.Module(body=[guard, device], type_ignores=[]), name, "exec")
    for available in (False, True):
        namespace = {"torch": SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: available), device=lambda value: value)}
        if available:
            exec(setup, namespace)
            assert namespace["device"] == "cuda"
        else:
            with pytest.raises(RuntimeError, match="GPU가 없습니다"):
                exec(setup, namespace)
            assert "device" not in namespace


def test_distributed_notebooks_pass_preflight():
    assert all(check_notebook(NOTEBOOKS / name) == [] for name in EXPECTED)


def test_preflight_accepts_legacy_student_task_tag(tmp_path):
    path = minimal_notebook(tmp_path, "01_gpu_inference.ipynb")
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        cell.metadata.tags = ["student-task"]
    nbformat.write(notebook, path)
    assert check_notebook(path) == []


@pytest.mark.parametrize("name", EXPECTED)
def test_preflight_accepts_definitions_without_executing_them(tmp_path, name):
    assert check_notebook(minimal_notebook(tmp_path, name)) == []


@pytest.mark.parametrize("body, expected", [
    ("def wrong_name():\n    return None", "함수가 아직 없습니다"),
    ("def {function}():\n    pass", "pass·설명만"),
    ('def {function}():\n    "Only a docstring"', "pass·설명만"),
    ("def {function}(:\n    pass", "문법 오류"),
])
def test_preflight_rejects_missing_or_unimplemented_functions(tmp_path, body, expected):
    name = "01_gpu_inference.ipynb"
    path = minimal_notebook(tmp_path, name)
    notebook = nbformat.read(path, as_version=4)
    function = notebook.cells[0].metadata.expected_function
    notebook.cells[0].source = body.format(function=function)
    nbformat.write(notebook, path)
    assert any(expected in issue for issue in check_notebook(path))


def test_preflight_checks_non_task_cell_syntax(tmp_path):
    path = minimal_notebook(tmp_path, "02_gpu_finetuning.ipynb")
    notebook = nbformat.read(path, as_version=4)
    notebook.cells.append(nbformat.v4.new_code_cell("if True\n    pass"))
    nbformat.write(notebook, path)
    assert any("문법 오류" in issue for issue in check_notebook(path))


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_function"])
def test_preflight_rejects_missing_duplicate_or_mismatched_metadata(tmp_path, mutation):
    path = minimal_notebook(tmp_path, "02_gpu_finetuning.ipynb")
    notebook = nbformat.read(path, as_version=4)
    if mutation == "missing":
        notebook.cells[0].metadata.clear()
    elif mutation == "duplicate":
        notebook.cells.append(nbformat.v4.new_code_cell(
            notebook.cells[0].source, metadata=dict(notebook.cells[0].metadata)))
    else:
        notebook.cells[0].metadata.expected_function = "wrong_name"
    nbformat.write(notebook, path)
    assert check_notebook(path)


@pytest.mark.parametrize("name", EXPECTED)
def test_check_cells_explain_missing_function_before_using_model(name):
    notebook = read_notebook(name)
    for cell in notebook.cells:
        if "exercise-check" not in cell.metadata.get("tags", []):
            continue
        function = EXPECTED[name][cell.metadata.exercise_id]
        namespace = {"TRAINING_ALLOWED": True, "HEAD_STAGE_COMPLETE": True,
                     "EXERCISE_CHECKS": {"02-step": True, "02-scope": True},
                     "INFERENCE_CHECKS": {"01-infer": True, "01-topk": True}}
        with pytest.raises(RuntimeError, match=function):
            exec(cell.source, namespace)


def test_missing_training_function_blocks_later_execution_and_writes(tmp_path):
    notebook = read_notebook("02_gpu_finetuning.ipynb")
    start = next(i for i, cell in enumerate(notebook.cells)
                 if cell.metadata.get("exercise_id") == "02-step"
                 and "exercise-check" in cell.metadata.get("tags", []))
    namespace = {"OUTPUT_DIR": tmp_path, "TRAINING_ALLOWED": True,
                 "EXERCISE_CHECKS": {"02-step": False, "02-scope": False},
                 "HEAD_STAGE_COMPLETE": False, "FINETUNE_SETUP_COMPLETE": False,
                 "FINETUNE_STAGE_COMPLETE": False, "MODEL_RELOADED": False,
                 "TABLES_SAVED": False, "REPORT_READY": False}
    for cell in notebook.cells[start:]:
        if cell.cell_type != "code" or "provided-function" in cell.metadata.get("tags", []):
            continue
        with pytest.raises(RuntimeError):
            exec(cell.source, namespace)
    assert list(tmp_path.iterdir()) == []
    assert "report" not in namespace


def test_inference_outputs_require_both_successful_checks(tmp_path):
    notebook = read_notebook("01_gpu_inference.ipynb")
    outputs = [cell.source for cell in notebook.cells if cell.cell_type == "code"
               and ("fig.savefig" in cell.source or "inference_report =" in cell.source)]
    assert len(outputs) == 2
    for checks in ({}, {"01-infer": True, "01-topk": False},
                   {"01-infer": False, "01-topk": True}):
        namespace = {"OUTPUT_DIR": tmp_path, "INFERENCE_CHECKS": checks}
        for source in outputs:
            with pytest.raises(RuntimeError, match="두 함수의 확인 셀"):
                exec(source, namespace)
        assert "inference_report" not in namespace
    assert list(tmp_path.iterdir()) == []


def test_training_results_require_actual_finetuning_completion(tmp_path):
    cells = read_notebook("02_gpu_finetuning.ipynb").cells
    loop_index = next(i for i, cell in enumerate(cells) if cell.cell_type == "code"
                      and "FINETUNE_STAGE_COMPLETE = True" in cell.source)
    output_cells = [cell for cell in cells[loop_index + 1:] if cell.cell_type == "code"]
    assert len(output_cells) >= 5
    namespace = {"OUTPUT_DIR": tmp_path, "TRAINING_ALLOWED": True,
                 "FINETUNE_STAGE_COMPLETE": False, "MODEL_RELOADED": True,
                 "TABLES_SAVED": True, "REPORT_READY": True}
    for cell in output_cells:
        with pytest.raises(RuntimeError, match="실습 확인과 실제 학습"):
            exec(cell.source, namespace)
    assert list(tmp_path.iterdir()) == []
    assert "report" not in namespace


@pytest.mark.parametrize("marker, flag, message", [
    ('with (OUTPUT_DIR / "training.csv").open', "MODEL_RELOADED", "모델 저장·재로딩"),
    ('report = {', "MODEL_RELOADED", "모델 저장·재로딩"),
    ('report = {', "TABLES_SAVED", "결과 표 저장"),
    ('artifact_names = [', "REPORT_READY", "보고서 준비가 완료되지"),
])
def test_training_save_and_report_steps_cannot_skip_prerequisites(tmp_path, marker, flag, message):
    source = next(cell.source for cell in read_notebook("02_gpu_finetuning.ipynb").cells
                  if cell.cell_type == "code" and marker in cell.source)
    namespace = {"OUTPUT_DIR": tmp_path, "TRAINING_ALLOWED": True,
                 "FINETUNE_STAGE_COMPLETE": True, "MODEL_RELOADED": True,
                 "TABLES_SAVED": True, "REPORT_READY": True}
    namespace[flag] = False
    with pytest.raises(RuntimeError, match=message):
        exec(source, namespace)
    assert list(tmp_path.iterdir()) == []
    assert "report" not in namespace
