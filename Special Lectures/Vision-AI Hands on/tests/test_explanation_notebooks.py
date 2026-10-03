"""Keep every source cell traceable without executing cloud code in the guides."""
import ast
from collections import Counter
import hashlib
from pathlib import Path
import re

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('00_hf_download_and_data.ipynb', '01_gpu_inference.ipynb', '02_gpu_finetuning.ipynb')
STDLIB_EXAMPLES = {'math', 'json', 'hashlib', 'random', 'collections', 'copy',
                   'pathlib', 'csv', 'io', 'statistics', 'itertools', 'types'}


def read_pair(name):
    source_path = ROOT / 'hf_colab_gpu/notebooks' / name
    guide_path = ROOT / 'hf_colab_gpu/explanations' / name.replace('.ipynb', '_explained.ipynb')
    return source_path, nbformat.read(source_path, as_version=4), guide_path, nbformat.read(guide_path, as_version=4)


@pytest.mark.parametrize('name', SOURCES)
def test_every_code_cell_has_an_exact_reading_copy_and_explanation(name):
    source_path, source, guide_path, guide = read_pair(name)
    nbformat.validate(guide)
    assert guide.metadata.workshop.mode == 'explanation'
    assert (guide_path.parent / guide.metadata.workshop.source_notebook).resolve() == source_path
    expected = {i: c for i, c in enumerate(source.cells) if c.cell_type == 'code'}
    references, explanations = [], []
    for cell in guide.cells:
        if 'source_reference' in cell.metadata:
            assert cell.cell_type == 'markdown', 'Source code must not run in the reading notebook'
            reference = cell.metadata.source_reference
            index = reference.cell_index
            assert index in expected
            assert (guide_path.parent / reference.notebook).resolve() == source_path
            code = expected[index].source
            assert reference.sha256 == hashlib.sha256(code.encode()).hexdigest()
            quoted = re.findall(r'```python\n(.*?)\n```', cell.source, flags=re.DOTALL)
            assert code in quoted, f'Missing exact source in {name} cell {index}'
            assert reference.kind == ('blank_exercise' if not code else 'code')
            references.append(index)
        if 'explains_source_cell' in cell.metadata:
            assert cell.cell_type == 'markdown'
            explanations.append(cell.metadata.explains_source_cell)
    assert Counter(references) == Counter(expected.keys())
    assert Counter(explanations) == Counter(expected.keys())
    assert references == list(expected), 'Reading order must follow the source notebook'


@pytest.mark.parametrize('name', SOURCES)
def test_only_small_cpu_examples_are_executable(name):
    _, source, _, guide = read_pair(name)
    original_sources = {cell.source for cell in source.cells if cell.cell_type == 'code' and cell.source}
    exercises = [cell for cell in guide.cells if cell.cell_type == 'code']
    assert exercises
    example_ids = []
    for cell in exercises:
        assert 'learning-example' in cell.metadata.get('tags', [])
        assert cell.source.strip() and cell.source not in original_sources
        example_ids.append(cell.metadata.example_id)
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split('.')[0] in STDLIB_EXAMPLES for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert (node.module or '').split('.')[0] in STDLIB_EXAMPLES
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {'open', 'exec', 'eval', 'compile', '__import__'}
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {'read_text', 'read_bytes', 'write_text', 'write_bytes',
                                              'open', 'mkdir', 'unlink', 'rename', 'rmdir'}
    assert len(example_ids) == len(set(example_ids))


@pytest.mark.parametrize('name', SOURCES)
def test_learning_examples_execute_in_order_without_cloud_libraries(name, capsys):
    _, _, _, guide = read_pair(name)
    namespace = {}
    for cell in guide.cells:
        if cell.cell_type != 'code':
            continue
        exec(compile(cell.source, f'{name}:{cell.metadata.example_id}', 'exec'), namespace)
        output = capsys.readouterr().out
        saved = ''.join(item.get('text', '') for item in cell.outputs if item.output_type == 'stream' and item.get('name') == 'stdout')
        assert output == saved, 'Saved example output is missing or stale'
        assert cell.execution_count is not None
        assert not any(item.output_type == 'error' for item in cell.outputs)
