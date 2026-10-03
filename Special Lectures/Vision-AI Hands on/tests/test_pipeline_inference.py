"""Exercise the pipeline entry point without model downloads or a GPU account."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import nbformat
from PIL import Image
import pytest

from hf_colab_gpu import pipeline_inference as inference


def test_hub_model_and_commit_reach_pipeline_with_explicit_processor():
    calls = {}
    processor = object()

    def processor_factory(*args, **kwargs):
        calls["processor"] = (args, kwargs)
        return processor

    def factory(*args, **kwargs):
        calls["pipeline"] = (args, kwargs)
        return "classifier"

    assert inference.create_classifier(
        inference.MODEL_ID, inference.MODEL_REVISION, "cuda:0",
        pipeline_factory=factory, processor_factory=processor_factory,
    ) == "classifier"
    assert calls["processor"] == ((inference.MODEL_ID,), {
        "revision": inference.MODEL_REVISION, "use_fast": False, "trust_remote_code": False,
    })
    args, settings = calls["pipeline"]
    assert args == ("image-classification",)
    assert settings == {
        "model": inference.MODEL_ID, "revision": inference.MODEL_REVISION,
        "image_processor": processor, "framework": "pt", "device": "cuda:0",
        "trust_remote_code": False,
        "model_kwargs": {"weights_only": True, "attn_implementation": "eager", "use_safetensors": False},
    }


def test_device_never_silently_falls_back_to_cpu():
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(RuntimeError, match="GPU가 없습니다"):
        inference.select_device(torch, "cuda")
    assert inference.select_device(torch, "cpu") == ("cpu", "CPU")
    torch.cuda.is_available = lambda: True
    torch.cuda.get_device_name = lambda index: "Test GPU"
    assert inference.select_device(torch, "cuda") == ("cuda:0", "Test GPU")


@pytest.mark.parametrize("predictions", [
    [], [{"label": "cup", "score": float("nan")}],
    [{"label": "cup", "score": float("inf")}],
    [{"label": "cup", "score": 1.2}], [{"label": "cup", "score": -0.1}],
    [{"label": "cup", "score": True}], [{"label": "", "score": 0.6}],
    [{"label": "cup", "score": "0.6"}],
])
def test_invalid_predictions_are_not_a_completed_result(predictions):
    with pytest.raises(ValueError):
        inference.validate_predictions(predictions, 1)


def test_unsorted_predictions_are_rejected():
    with pytest.raises(ValueError, match="내림차순"):
        inference.validate_predictions([
            {"label": "first", "score": 0.1}, {"label": "second", "score": 0.8},
        ], 2)


def test_new_result_does_not_overwrite_old_file(tmp_path):
    path = tmp_path / "result.json"
    inference.save_new_json(path, {"existing": True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        inference.save_new_json(path, {"new": True})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_end_to_end_result_tracks_original_image_and_actual_predictions(tmp_path, monkeypatch, capsys):
    image_path = tmp_path / "input.png"
    Image.new("L", (4, 4), 90).save(image_path)
    output = tmp_path / "output.json"
    original_bytes = image_path.read_bytes()
    calls = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(inference, "version", lambda name: f"test-{name}")

    class Classifier:
        model = SimpleNamespace(config=SimpleNamespace(num_labels=1000))

        def __call__(self, image, top_k):
            calls.append((image.mode, top_k))
            return [{"label": "test label", "score": 0.75}]

    monkeypatch.setattr(inference, "create_classifier", lambda *args: Classifier())
    args = inference.parse_args([
        "--image", str(image_path), "--output", str(output),
        "--device", "cpu", "--run-id", "test-run", "--top-k", "1",
    ])
    report = inference.run_inference(args)
    assert json.loads(output.read_text()) == report
    assert report["image_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert report["status"] == "completed" and report["run_id"] == "test-run"
    assert report["task"] == "image-classification"
    assert report["model_id"] == inference.MODEL_ID
    assert report["model_revision"] == inference.MODEL_REVISION
    assert report["device"] == "cpu" and report["device_name"] == "CPU"
    assert report["predictions"] == [{"label": "test label", "score": 0.75}]
    assert set(report["versions"]) == {"torch", "transformers", "huggingface-hub"}
    assert calls == [("RGB", 1)]
    assert "HF_PIPELINE_INFERENCE_COMPLETE" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        inference.run_inference(args)
    assert "HF_PIPELINE_INFERENCE_COMPLETE" not in capsys.readouterr().out


def test_missing_cuda_stops_before_model_loading(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setattr(inference, "create_classifier", lambda *args: pytest.fail("No model loading"))
    output = tmp_path / "result.json"
    args = inference.parse_args(["--image", "unused.png", "--output", str(output)])
    with pytest.raises(RuntimeError, match="GPU가 없습니다"):
        inference.run_inference(args)
    assert not output.exists()


def test_pipeline_demo_precedes_two_provided_functions():
    path = Path(__file__).resolve().parents[1] / "hf_colab_gpu/notebooks/01_gpu_inference.ipynb"
    notebook = nbformat.read(path, as_version=4)
    provided = [i for i, cell in enumerate(notebook.cells)
                if cell.cell_type == "code" and "provided-pipeline-demo" in cell.metadata.get("tags", [])]
    functions = [i for i, cell in enumerate(notebook.cells)
                 if "provided-function" in cell.metadata.get("tags", [])]
    assert len(provided) == 1 and len(functions) == 2 and provided[0] < min(functions)
    assert [notebook.cells[index].metadata.expected_function for index in functions] == [
        "infer_probabilities", "topk_predictions",
    ]
    for index in functions:
        cell = notebook.cells[index]
        assert cell.cell_type == "code" and cell.source.strip()
        assert any(isinstance(node, ast.FunctionDef) and node.name == cell.metadata.expected_function
                   for node in ast.parse(cell.source).body)
    source = notebook.cells[provided[0]].source
    assert "model=model, image_processor=processor" in source
    assert "pipeline_predictions" in source
