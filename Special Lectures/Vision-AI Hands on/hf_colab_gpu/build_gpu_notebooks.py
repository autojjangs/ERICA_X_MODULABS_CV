"""Generate student notebooks with all PyTorch/Transformers code in visible cells."""
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

HERE = Path(__file__).resolve().parent
MODEL_ID = "facebook/deit-tiny-patch16-224"
REVISION = "b3428f18dcc7b543470d07f14b4a4157815d1880"


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    source = dedent(text).strip()
    if len(source.splitlines()) > 40:
        raise ValueError(f"Code cell exceeds 40 lines: {source[:80]}")
    return nbf.v4.new_code_cell(source)


def setup_cells():
    return [md("""
    ## 1. 라이브러리와 GPU 확인

    이 노트북은 **Colab CLI로 연결한 GPU 세션**에서 실행합니다. Codespaces는 파일 작성·업로드에 사용합니다.
    `00_hf_download_and_data.ipynb`에서 받은 모델 파일과 기존 `data/prepared`의 데이터를 먼저 업로드하세요.
    GPU가 없으면 여기서 멈춥니다. 설치와 세션 연결 순서는 이 폴더의 README를 따릅니다.
    """), code('''
    TRAINING_ALLOWED = False
    import os
    import sys
    import json
    import hashlib
    import random
    from pathlib import Path
    from importlib.metadata import version

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    from IPython import get_ipython
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic("matplotlib", "inline")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU가 없습니다. Colab CLI GPU 세션으로 실행하세요.")
    device = torch.device("cuda")
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    plt.rcParams.update({"figure.dpi": 110, "font.size": 11})
    print("Python:", sys.executable)
    print("GPU:", torch.cuda.get_device_name(0))
    print({name: version(name) for name in ("torch", "transformers", "huggingface-hub")})
    '''), code(f'''
    candidates = [Path.cwd(), *Path.cwd().parents, Path("/content/vision-ai")]
    ROOT = next((p for p in candidates if (p / ".vision-lab-root").is_file()), None)
    if ROOT is None:
        raise FileNotFoundError(".vision-lab-root와 실습 파일을 먼저 업로드하세요.")
    MODEL_ID = {MODEL_ID!r}
    MODEL_REVISION = {REVISION!r}
    MODEL_DIR = ROOT / "hf_colab_gpu/models/deit-tiny"
    DATA_DIR = ROOT / "data/prepared"
    OUTPUT_DIR = ROOT / "hf_colab_gpu/results/gpu"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def sha256_file(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    download_manifest = json.loads((MODEL_DIR / "download_manifest.json").read_text())
    assert download_manifest["model_id"] == MODEL_ID
    assert download_manifest["revision"] == MODEL_REVISION
    for name in ("config.json", "preprocessor_config.json", "pytorch_model.bin"):
        assert sha256_file(MODEL_DIR / name) == download_manifest["files"][name], name
    print("HF CLI 다운로드 파일 확인:", MODEL_ID, MODEL_REVISION[:12])
    '''), md("""
    ## 2. 데이터와 분할 확인

    CIFAR-100에서 bottle·bowl·can·cup·plate를 골랐습니다. 원본은 32×32 이미지이며 모델 입력 크기로 확대합니다.
    실제 작업대 사진과는 차이가 있으므로 이 결과를 현장 정확도로 해석하지 않습니다.
    학습 500장, 검증 100장, 최종 평가 200장으로 나눴고, 학습 중 모델 선택에는 검증 데이터만 사용합니다.
    """), code('''
    manifest_file = DATA_DIR / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    classes = manifest["classes"]
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("클래스 목록을 확인하세요.")
    identity = {k: manifest[k] for k in ("classes", "splits", "seed", "preprocess")}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    assert fingerprint == manifest["dataset_sha256"]
    splits, seen_ids = {}, set()
    for split in ("train", "validation", "test"):
        record = manifest["splits"][split]
        assert record["file"] == f"{split}.npz"
        file = DATA_DIR / record["file"]
        assert sha256_file(file) == record["sha256"], split
        with np.load(file, allow_pickle=False) as stored:
            images, labels, ids = stored["images"], stored["labels"], stored["ids"]
        assert images.dtype == np.uint8 and images.ndim == 4 and images.shape[-1] == 3
        assert labels.ndim == 1 and np.issubdtype(labels.dtype, np.integer)
        assert len(images) == len(labels) == len(ids) == record["count"] > 0
        assert labels.min() >= 0 and labels.max() < len(classes)
        assert len(set(ids)) == len(ids) and not seen_ids.intersection(ids)
        seen_ids.update(ids)
        splits[split] = {"images": images, "labels": labels, "ids": ids}
        print(split, len(labels), "images")
    print("클래스 순서:", classes)
    '''), md("""
    ## 3. Transformers로 모델 불러오기

    `AutoImageProcessor`가 크기 조정·정규화를 맡고, `AutoModelForImageClassification`이 사전학습 모델을 만듭니다.
    `hf download`로 받은 로컬 파일만 사용하므로 이 단계에서 추가 다운로드는 하지 않습니다.
    원본 가중치는 `.bin`이며 `weights_only=True`로 읽습니다. 실습 후 저장할 모델은 Safetensors 형식입니다.
    """), code('''
    processor = AutoImageProcessor.from_pretrained(
        MODEL_DIR, local_files_only=True, use_fast=False,
    )
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_DIR, local_files_only=True, use_safetensors=False, weights_only=True,
        attn_implementation="eager",
    ).to(device)
    print(type(model).__name__, "· 원래 분류 수:", model.config.num_labels)
    print("전체 파라미터:", f"{sum(p.numel() for p in model.parameters()):,}")
    ''')]


def inference_cells():
    return [md("""
    # Hugging Face 모델을 Colab GPU에서 추론하기

    **목표:** Hugging Face CLI로 받은 모델을 Transformers로 불러오고, GPU에서 이미지 한 장의 분류 결과를 확인합니다.
    이 단계는 원래 모델의 1,000개 ImageNet 분류를 사용합니다. 다음 실습에서 우리 데이터의 5개 클래스로 바꿉니다.
    """), *setup_cells(), md("""
    ## 4. 이미지 한 장을 GPU에서 추론하기

    정답이 cup인 예시를 선택합니다. 입력 사진이 작고 원래 모델의 분류 목록도 다르므로,
    이 한 장의 상위 예측만으로 실습 데이터 전체의 정확도를 판단하지 않습니다.
    """), code('''
    target_class = classes.index("cup") if "cup" in classes else 0
    image_index = int(np.flatnonzero(splits["test"]["labels"] == target_class)[0])
    image = Image.fromarray(splits["test"]["images"][image_index]).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    model.eval()
    with torch.inference_mode():
        probabilities = model(**inputs).logits.softmax(dim=-1)[0]
    values, indices = probabilities.topk(5)
    predictions = [
        {"label": model.config.id2label[int(index)], "probability": float(value)}
        for value, index in zip(values.cpu(), indices.cpu())
    ]
    print("실습 데이터 정답:", classes[target_class])
    for item in predictions:
        print(f"{item['probability']:6.2%}  {item['label']}")
    '''), code('''
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    axes[0].imshow(image)
    axes[0].set_title(f"Input: {classes[target_class]} (32×32)")
    axes[0].axis("off")
    labels = [item["label"].split(",")[0] for item in predictions]
    axes[1].barh(labels[::-1], [p["probability"] for p in predictions][::-1], color="#1a73e8")
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Pretrained ImageNet: top 5")
    fig.savefig(OUTPUT_DIR / "pretrained_top5.png", dpi=150, bbox_inches="tight")
    plt.show()
    '''), md("""
    ## 5. 실행 기록 확인

    예측 결과와 실제 GPU 이름을 저장합니다. 이 결과는 원래 모델을 불러오는 데 성공했는지 확인하는 기록입니다.
    다음 노트북에서는 분류 헤드만 학습한 기준 모델과, 마지막 블록까지 파인튜닝한 모델을 비교합니다.
    """), code('''
    inference_report = {
        "status": "completed", "platform": "gpu", "device": torch.cuda.get_device_name(0),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "download_manifest_sha256": sha256_file(MODEL_DIR / "download_manifest.json"),
        "dataset_sha256": manifest["dataset_sha256"],
        "sample_id": str(splits["test"]["ids"][image_index]),
        "ground_truth": classes[target_class], "top5": predictions,
        "versions": {name: version(name) for name in ("torch", "transformers", "huggingface-hub")},
    }
    (OUTPUT_DIR / "inference_report.json").write_text(
        json.dumps(inference_report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print("HF_GPU_INFERENCE_COMPLETE", OUTPUT_DIR / "inference_report.json")
    ''')]


def training_cells():
    cells = [md("""
    # 작업대 물체 분류 모델을 GPU에서 파인튜닝하기

    **목표:** Transformers가 불러온 모델의 분류 헤드를 학습한 뒤, 마지막 Transformer 블록도 조금 수정해 성능을 비교합니다.
    PyTorch의 `DataLoader`, `CrossEntropyLoss`, `Adam`과 학습 반복문을 직접 읽습니다.
    모델 구조는 Transformers로 불러오고 학습 코드는 PyTorch로 작성합니다.

    GPU 사용 여부, 검증 데이터로 선택한 모델, 고정한 가중치의 변화, 저장한 모델의 재로딩까지 확인합니다.
    정확도 상승은 보장하지 않습니다. 데이터와 실행 환경에 따라 달라지는 실제 결과를 기록하세요.
    """), *setup_cells(), md("""
    ## 4. 학습 설정과 미니배치 만들기

    배치 크기는 16, 학습은 단계별 3 epoch입니다. 이미지 전처리는 Transformers에 맡깁니다.
    파일을 덮어쓰지 않도록 이전 `report.json`이 있으면 멈춥니다. 재실행할 때는 이전 결과 폴더를 보관한 뒤 새로 시작하세요.
    """), code('''
    TRAINING_ALLOWED = False
    if (OUTPUT_DIR / "report.json").exists():
        raise FileExistsError("완료된 결과가 있습니다. results/gpu를 보관한 뒤 재실행하세요.")
    import csv
    import time
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    BATCH_SIZE = 16
    HEAD_EPOCHS, FINETUNE_EPOCHS = 3, 3
    HEAD_LR, FINETUNE_LR = 1e-3, 1e-4
    started = time.perf_counter()
    loaders = {}
    for name, split in splits.items():
        pil_images = [Image.fromarray(pixels).convert("RGB") for pixels in split["images"]]
        pixels = processor(images=pil_images, return_tensors="pt")["pixel_values"]
        dataset = TensorDataset(pixels, torch.tensor(split["labels"], dtype=torch.long))
        loaders[name] = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=(name == "train"),
            generator=torch.Generator().manual_seed(SEED), num_workers=0,
        )
    print("입력 크기:", tuple(loaders["train"].dataset.tensors[0].shape))
    TRAINING_ALLOWED = True
    '''), md("""
    ## 5. 분류 헤드만 학습하도록 설정하기

    1,000개 클래스를 출력하던 마지막 선형층을 실습 클래스 수에 맞춰 교체합니다.
    나머지 가중치는 `requires_grad=False`로 고정합니다. 모델의 전체 구조를 다시 구현할 필요가 없습니다.
    """), code('''
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.classifier = nn.Linear(model.config.hidden_size, len(classes)).to(device)
    nn.init.normal_(model.classifier.weight, std=model.config.initializer_range)
    nn.init.zeros_(model.classifier.bias)
    model.num_labels = len(classes)
    model.config.num_labels = len(classes)
    model.config.id2label = dict(enumerate(classes))
    model.config.label2id = {label: index for index, label in enumerate(classes)}
    criterion = nn.CrossEntropyLoss()

    def parameter_sha256(named_parameters):
        digest = hashlib.sha256()
        for name, parameter in named_parameters:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    backbone_before = parameter_sha256(model.vit.named_parameters())
    history = []
    print("헤드 학습 파라미터:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    '''), md("""
    ## 6. 평가 함수 만들기

    이 함수는 손실·정확도·예측값을 돌려줍니다. 학습 중에는 검증 데이터만 넣습니다.
    최종 평가 데이터는 각 단계의 최적 모델을 고른 뒤에 사용합니다.
    """), code('''
    def evaluate(data_loader):
        model.eval()
        loss_sum, all_labels, all_predictions = 0.0, [], []
        with torch.inference_mode():
            for pixels, labels in data_loader:
                pixels, labels = pixels.to(device), labels.to(device)
                logits = model(pixel_values=pixels).logits
                loss_sum += criterion(logits, labels).item() * len(labels)
                all_labels.extend(labels.cpu().tolist())
                all_predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        labels = np.asarray(all_labels)
        predictions = np.asarray(all_predictions)
        matrix = np.zeros((len(classes), len(classes)), dtype=int)
        np.add.at(matrix, (labels, predictions), 1)
        denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
        f1 = np.divide(2 * matrix.diagonal(), denominator,
                       out=np.zeros(len(classes)), where=denominator > 0)
        return {
            "loss": loss_sum / len(labels), "accuracy": float(np.mean(labels == predictions)),
            "macro_f1": float(f1.mean()), "count": len(labels),
            "confusion_matrix": matrix.tolist(), "predictions": predictions.tolist(),
        }
    '''), md("""
    ## 7. 기준 모델: 분류 헤드 학습

    `zero_grad → forward → loss → backward → step`이 한 배치의 학습 순서입니다.
    검증 정확도가 가장 높은 epoch를 선택하고, 동률이면 검증 손실이 낮은 모델을 고릅니다.
    """), code('''
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=HEAD_LR)
    best_key, best_head, best_head_epoch = (-1.0, float("-inf")), None, None
    for epoch in range(1, HEAD_EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_count = 0.0, 0, 0
        for pixels, labels in loaders["train"]:
            pixels, labels = pixels.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values=pixels).logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_correct += int((logits.argmax(-1) == labels).sum())
            train_count += len(labels)
        validation = evaluate(loaders["validation"])
        history.append({"stage": "head", "epoch": epoch, "train_loss": train_loss / train_count,
                        "train_accuracy": train_correct / train_count,
                        "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"]})
        key = (validation["accuracy"], -validation["loss"])
        if key > best_key:
            best_key, best_head_epoch = key, epoch
            best_head = {name: value.detach().cpu().clone()
                         for name, value in model.classifier.state_dict().items()}
        print(f"head {epoch}/{HEAD_EPOCHS} · validation {validation['accuracy']:.1%}")
    model.classifier.load_state_dict(best_head)
    assert parameter_sha256(model.vit.named_parameters()) == backbone_before
    head_validation = evaluate(loaders["validation"])
    head_test = evaluate(loaders["test"])
    print(f"선택 epoch {best_head_epoch} · head test {head_test['accuracy']:.1%}")
    '''), md("""
    ## 8. 마지막 블록까지 파인튜닝하기

    앞의 11개 블록은 계속 고정합니다. 마지막 블록·마지막 LayerNorm·분류 헤드만 학습 대상으로 바꿉니다.
    방금 학습한 헤드에서 이어서 시작하고, 학습률은 0.0001로 낮춥니다.
    """), code('''
    last_block_index = len(model.vit.encoder.layer) - 1
    last_block_prefix = f"vit.encoder.layer.{last_block_index}."
    for module in (model.vit.encoder.layer[-1], model.vit.layernorm, model.classifier):
        for parameter in module.parameters():
            parameter.requires_grad = True
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    frozen_before = parameter_sha256((n, p) for n, p in model.named_parameters() if not p.requires_grad)
    tail_before = parameter_sha256(model.vit.encoder.layer[-1].named_parameters())
    trainable_count = sum(parameter.numel() for parameter in trainable.values())
    assert all(name.startswith((last_block_prefix, "vit.layernorm.", "classifier.")) for name in trainable)
    print("파인튜닝 파라미터:", f"{trainable_count:,}")
    optimizer = torch.optim.Adam(trainable.values(), lr=FINETUNE_LR)
    best_key, best_tail, best_finetune_epoch = (-1.0, float("-inf")), None, None
    '''), code('''
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_count = 0.0, 0, 0
        for pixels, labels in loaders["train"]:
            pixels, labels = pixels.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values=pixels).logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_correct += int((logits.argmax(-1) == labels).sum())
            train_count += len(labels)
        validation = evaluate(loaders["validation"])
        history.append({"stage": "finetune", "epoch": epoch, "train_loss": train_loss / train_count,
                        "train_accuracy": train_correct / train_count,
                        "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"]})
        key = (validation["accuracy"], -validation["loss"])
        if key > best_key:
            best_key, best_finetune_epoch = key, epoch
            best_tail = {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()}
        print(f"finetune {epoch}/{FINETUNE_EPOCHS} · validation {validation['accuracy']:.1%}")
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(best_tail[name].to(device))
    finetune_validation = evaluate(loaders["validation"])
    finetune_test = evaluate(loaders["test"])
    frozen_after = parameter_sha256((n, p) for n, p in model.named_parameters() if not p.requires_grad)
    tail_after = parameter_sha256(model.vit.encoder.layer[-1].named_parameters())
    assert frozen_before == frozen_after, "고정한 가중치가 바뀌었습니다."
    assert tail_before != tail_after, "마지막 블록의 가중치가 바뀌지 않았습니다."
    print(f"선택 epoch {best_finetune_epoch} · fine-tune test {finetune_test['accuracy']:.1%}")
    '''), md("""
    ## 9. 학습 전후 비교하기

    Head는 분류 헤드만 학습한 기준 모델, Fine-tune은 마지막 블록까지 학습한 모델입니다.
    학습 곡선에서는 검증 성능을, 혼동행렬에서는 마지막에 남겨 둔 평가 이미지의 오분류를 확인합니다.
    두 단계 모두 같은 200장으로 평가하므로 단순히 숫자가 올랐는지뿐 아니라 어떤 클래스가 달라졌는지도 살펴보세요.
    """), code('''
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for stage, color in (("head", "#1a73e8"), ("finetune", "#188038")):
        rows = [row for row in history if row["stage"] == stage]
        epochs = [row["epoch"] for row in rows]
        axes[0].plot(epochs, [r["train_loss"] for r in rows], "o-", color=color, label=f"{stage}: train")
        axes[0].plot(epochs, [r["validation_loss"] for r in rows], "s--", color=color, label=f"{stage}: validation")
        axes[1].plot(epochs, [r["validation_accuracy"] for r in rows], "o-", color=color, label=stage)
    axes[0].set(title="Loss by training stage", xlabel="Epoch within each stage", ylabel="Cross-entropy loss")
    axes[1].set(title="Validation accuracy (100 images)", xlabel="Epoch within each stage", ylabel="Accuracy", ylim=(0, 1))
    for axis in axes:
        axis.legend(fontsize=9)
        axis.grid(alpha=0.2)
        axis.set_xticks(range(1, max(HEAD_EPOCHS, FINETUNE_EPOCHS) + 1))
    fig.savefig(OUTPUT_DIR / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    '''), code('''
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    maximum = max(np.max(head_test["confusion_matrix"]), np.max(finetune_test["confusion_matrix"]))
    for axis, title, metrics in zip(axes, ("Head", "Fine-tune"), (head_test, finetune_test)):
        matrix = np.asarray(metrics["confusion_matrix"])
        image = axis.imshow(matrix, vmin=0, vmax=maximum, cmap="Blues")
        axis.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
        axis.set_yticks(range(len(classes)), classes)
        axis.set(title=f"{title}: {metrics['accuracy']:.1%} ({metrics['count']} test images)",
                 xlabel="Predicted class", ylabel="True class")
        for row, column in np.ndindex(matrix.shape):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] > maximum / 2 else "black")
    fig.colorbar(image, ax=axes, label="Image count", shrink=0.85)
    fig.savefig(OUTPUT_DIR / "confusion_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"정확도 변화: {head_test['accuracy']:.1%} → {finetune_test['accuracy']:.1%}")
    print(f"Macro F1 변화: {head_test['macro_f1']:.3f} → {finetune_test['macro_f1']:.3f}")
    '''), md("""
    ## 10. 모델 저장 후 다시 불러오기

    `save_pretrained()`가 모델 설정과 Safetensors 가중치를 함께 저장합니다.
    저장된 폴더를 `from_pretrained()`로 다시 불러와 같은 이미지의 출력이 일치하는지 확인합니다.
    """), code('''
    saved_model_dir = OUTPUT_DIR / "finetuned_model"
    model.save_pretrained(saved_model_dir, safe_serialization=True)
    processor.save_pretrained(saved_model_dir)
    restored_processor = AutoImageProcessor.from_pretrained(saved_model_dir, local_files_only=True, use_fast=False)
    restored = AutoModelForImageClassification.from_pretrained(
        saved_model_dir, local_files_only=True, use_safetensors=True, attn_implementation="eager",
    ).to(device).eval()
    check_image = Image.fromarray(splits["test"]["images"][0]).convert("RGB")
    check_input = processor(images=check_image, return_tensors="pt").to(device)
    restored_input = restored_processor(images=check_image, return_tensors="pt").to(device)
    model.eval()
    with torch.inference_mode():
        before_save = model(**check_input).logits
        after_load = restored(**restored_input).logits
    torch.testing.assert_close(before_save, after_load, rtol=1e-5, atol=1e-6)
    assert restored.config.id2label == dict(enumerate(classes))
    reload_max_abs_diff = float((before_save - after_load).abs().max())
    print("저장·재로딩 검증 통과 · 최대 출력 차이:", reload_max_abs_diff)
    '''), md("""
    ## 11. 결과 파일과 실행 기록 저장

    `report.json`은 실제 장치·데이터·학습 설정·평가 결과를 담습니다.
    Colab 세션을 종료하기 전에 결과 폴더와 `finetuned_model`을 Codespaces로 내려받으세요.
    """), code('''
    with (OUTPUT_DIR / "training.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (OUTPUT_DIR / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id", "true_label", "head_prediction", "finetune_prediction"])
        for sample_id, label, head_pred, fine_pred in zip(
            splits["test"]["ids"], splits["test"]["labels"], head_test["predictions"], finetune_test["predictions"],
        ):
            writer.writerow([sample_id, classes[int(label)], classes[head_pred], classes[fine_pred]])
    def summary(metrics):
        return {key: value for key, value in metrics.items() if key != "predictions"}
    report = {
        "status": "completed", "platform": "gpu", "accelerator_verified": True,
        "device": torch.cuda.get_device_name(0), "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "download_manifest_sha256": sha256_file(MODEL_DIR / "download_manifest.json"),
        "dataset_manifest_sha256": sha256_file(manifest_file), "dataset_sha256": manifest["dataset_sha256"],
        "classes": classes, "split_counts": {k: len(v["labels"]) for k, v in splits.items()},
        "versions": {name: version(name) for name in ("torch", "transformers", "huggingface-hub")},
        "config": {"seed": SEED, "batch_size": BATCH_SIZE, "head_epochs": HEAD_EPOCHS,
                   "finetune_epochs": FINETUNE_EPOCHS, "head_lr": HEAD_LR, "finetune_lr": FINETUNE_LR,
                   "optimizer": "Adam", "checkpoint_selection": "validation_accuracy_then_loss"},
        "trainable_parameters": trainable_count,
        "stages": {"head": {"selected_epoch": best_head_epoch, "validation": summary(head_validation), "test": summary(head_test)},
                   "finetune": {"selected_epoch": best_finetune_epoch, "validation": summary(finetune_validation), "test": summary(finetune_test)}},
        "verification": {"frozen_parameters_unchanged": frozen_before == frozen_after,
                         "last_block_changed": tail_before != tail_after, "reload_max_abs_diff": reload_max_abs_diff,
                         "frozen_before": frozen_before, "frozen_after": frozen_after,
                         "last_block_before": tail_before, "last_block_after": tail_after},
        "checkpoint": {"path": "finetuned_model", "format": "safetensors",
                       "sha256": sha256_file(saved_model_dir / "model.safetensors")},
        "elapsed_seconds": time.perf_counter() - started,
    }
    '''), code('''
    artifact_names = [
        "training.csv", "predictions.csv", "learning_curves.png", "confusion_comparison.png",
        "finetuned_model/config.json", "finetuned_model/preprocessor_config.json",
        "finetuned_model/model.safetensors",
    ]
    report["artifacts"] = artifact_names
    report["artifact_sha256"] = {
        name: sha256_file(OUTPUT_DIR / name) for name in artifact_names
    }
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("HF_GPU_FINETUNING_COMPLETE", OUTPUT_DIR / "report.json")
    '''), md("""
    ## 다음 실습

    결과 표와 오분류 이미지를 보고 데이터 수·촬영 조건·학습 범위를 어떻게 바꿀지 정해 보세요.
    저장된 `finetuned_model` 폴더는 `AutoImageProcessor`와 `AutoModelForImageClassification`로 바로 다시 읽을 수 있습니다.
    [AI 코딩 실습 안내](../ai-coding-workshop.md)에 따라 바꿔 본 조건과 실행 결과를 정리해 보세요.
    """)]
    settings_seen = False
    for cell in cells:
        if cell.cell_type != "code":
            continue
        if settings_seen:
            cell.source = (
                'if not TRAINING_ALLOWED:\n'
                '    raise RuntimeError("학습 준비가 완료되지 않았습니다. 기존 결과를 보관한 뒤 처음부터 실행하세요.")\n'
                + cell.source
            )
            if len(cell.source.splitlines()) > 40:
                raise ValueError("Guarded code cell exceeds 40 lines")
        if 'if (OUTPUT_DIR / "report.json").exists():' in cell.source:
            settings_seen = True
    return cells


def build():
    from inference_exercises import adapt_inference
    from finetuning_exercises import adapt_training

    notebooks = {"01_gpu_inference.ipynb": adapt_inference(inference_cells()),
                 "02_gpu_finetuning.ipynb": adapt_training(training_cells())}
    for name, cells in notebooks.items():
        notebook = nbf.v4.new_notebook(cells=cells, metadata={
            "kernelspec": {"display_name": "Python 3 (Colab GPU)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "colab": {"name": name}, "accelerator": "GPU",
            "workshop": {"mode": "guided-code", "student_task_count": 0,
                         "provided_function_count": 2, "requires_completed_student_cells": False},
        })
        destination = HERE / "notebooks" / name
        if destination.exists():
            previous = nbf.read(destination, as_version=4)
            # Keep existing cell links stable when rebuilding the same walkthrough.
            if len(previous.cells) == len(cells):
                for old_cell, new_cell in zip(previous.cells, notebook.cells):
                    if old_cell.cell_type == new_cell.cell_type:
                        new_cell.id = old_cell.id
        nbf.validate(notebook)
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(notebook, destination)
        print(name, len(cells), "cells")


if __name__ == "__main__":
    build()
