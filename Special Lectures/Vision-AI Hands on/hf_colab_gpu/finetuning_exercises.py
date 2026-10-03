"""Insert two complete functions into the real GPU training walkthrough."""
from copy import deepcopy
from textwrap import dedent

import nbformat as nbf

GUARD = ('if not TRAINING_ALLOWED:\n'
         '    raise RuntimeError("학습 준비가 완료되지 않았습니다. 기존 결과를 보관한 뒤 처음부터 실행하세요.")\n')


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text, *, guarded=True, tag=None, exercise=None):
    source = (GUARD if guarded else '') + dedent(text).strip()
    if len(source.splitlines()) > 40:
        raise ValueError(f"Exercise code exceeds 40 lines ({len(source.splitlines())}): {source[:100]}")
    metadata = {}
    if tag:
        metadata['tags'] = [tag]
    if exercise:
        metadata['exercise_id'] = exercise
    return nbf.v4.new_code_cell(source, metadata=metadata)


def provided_function(identifier, function, source):
    cell = code(source, tag='provided-function', exercise=identifier)
    cell.metadata['expected_function'] = function
    return cell


def step_exercise():
    return [md('''
    ## 7. 실습 1 · 한 배치의 학습 과정 살펴보기

    손실을 계산하고 모델의 가중치를 실제로 한 번 바꾸는 완성된 함수를 살펴봅니다.
    먼저 작은 연습 모델로 확인하고, 통과한 **같은 함수**를 아래 실제 DeiT 학습 반복문에서 사용합니다.

    **제공 입력:** `model`, `optimizer`, `criterion`, 장치에 올라간 `pixels`와 `labels`입니다.
    모델은 `model(pixel_values=pixels).logits`로 점수를 반환합니다.

    **함수의 동작**

    1. `train_one_batch(model, optimizer, criterion, pixels, labels)`가 한 배치를 학습합니다.
    2. 학습 모드로 바꾸고 이전 기울기를 지운 다음, 예측 점수와 손실을 계산합니다.
    3. 역전파와 optimizer 업데이트를 각각 한 번 실행합니다.
    4. `(손실 float, 맞힌 수 int, 사진 수 int)` 튜플을 반환합니다.
       손실과 맞힌 수는 **이번 업데이트 전**에 구한 예측 점수로 계산합니다.
    5. 새 optimizer나 모델을 만들지 않습니다. 제공된 장치·학습률·학습 대상 가중치를 유지합니다.

    **연습 입력의 예상 결과:** 아래 2개 입력에서 첫 손실은 약 `0.6931`, 정답 수는 `1`,
    전체 수는 `2`입니다. 한 번 학습한 뒤에는 가중치가 달라집니다. 같은 입력으로 다시 학습하면
    이 작은 예제의 손실은 약 `0.6685`가 됩니다. 실제 사진 학습의 손실이나 정확도를 보장하는 값은 아닙니다.

    **선택: AI에게 설명 요청하기**

    > 저는 Python 초보자입니다. 아래 `train_one_batch` 함수가 한 배치를 학습하는 순서를 설명해 주세요.
    > train, zero_grad, forward, CrossEntropyLoss, backward, step이 각각 무슨 일을 하는지 알려 주세요.
    > 기울기를 매번 지우는 이유와 backward와 step의 차이도 쉬운 말로 설명해 주세요.
    > 반환하는 손실과 맞힌 수가 업데이트 전 예측을 기준으로 하는 이유도 알려 주세요.
    '''), code('''
    from types import SimpleNamespace
    EXERCISE_CHECKS["02-step"] = False

    class PracticeClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Linear(2, 2)

        def forward(self, pixel_values):
            return SimpleNamespace(logits=self.classifier(pixel_values))

    def make_practice_model():
        # 연습 모델 초기화가 본학습의 난수 순서를 바꾸지 않도록 합니다.
        with torch.random.fork_rng(devices=[]):
            practice = PracticeClassifier().to(device)
        with torch.no_grad():
            practice.classifier.weight.zero_()
            practice.classifier.bias.zero_()
        return practice.eval()

    practice_pixels = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
    practice_labels = torch.tensor([0, 1], dtype=torch.long, device=device)
    print("연습 입력:", practice_pixels.tolist(), "· 정답:", practice_labels.tolist())
    print("연습 optimizer: SGD, 학습률 0.1 / 실제 본학습: 기존 Adam 설정 유지")
    ''', tag='exercise-input', exercise='02-step'),
    md('아래 완성된 함수 셀을 실행하세요. 다음 확인 셀에서 두 번 학습한 결과를 봅니다.'),
    provided_function('02-step', 'train_one_batch', '''
    def train_one_batch(model, optimizer, criterion, pixels, labels):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(pixel_values=pixels).logits
        loss = criterion(logits, labels)
        # 이번 업데이트 전 예측으로 맞힌 수와 사진 수를 기록합니다.
        correct = int((logits.argmax(dim=-1) == labels).sum())
        count = int(len(labels))
        loss.backward()
        optimizer.step()
        return float(loss.item()), correct, count
    '''),
    code('''
    EXERCISE_CHECKS["02-step"] = False
    if not callable(globals().get("train_one_batch")):
        raise RuntimeError("앞의 train_one_batch 함수 셀을 먼저 실행하세요.")
    practice = make_practice_model()
    practice_optimizer = torch.optim.SGD(practice.parameters(), lr=0.1)
    first = train_one_batch(practice, practice_optimizer, criterion, practice_pixels, practice_labels)
    second = train_one_batch(practice, practice_optimizer, criterion, practice_pixels, practice_labels)
    for result in (first, second):
        assert isinstance(result, tuple) and len(result) == 3, "튜플 (손실, 맞힌 수, 사진 수)을 반환하세요."
        assert type(result[0]) is float and all(type(x) is int for x in result[1:])
    assert practice.training, "함수 안에서 학습 모드로 바꾸세요."
    assert abs(first[0] - 0.69314718) < 1e-5 and first[1:] == (1, 2)
    assert abs(second[0] - 0.66845965) < 1e-5 and second[1:] == (2, 2)
    expected_weight = torch.tensor([[0.04937513, -0.04937513], [-0.04937513, 0.04937513]], device=device)
    torch.testing.assert_close(practice.classifier.weight, expected_weight, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(practice.classifier.bias, torch.zeros(2, device=device), rtol=0, atol=1e-6)
    EXERCISE_CHECKS["02-step"] = True
    print(f"1회: loss={first[0]:.4f}, correct={first[1]}, count={first[2]}")
    print(f"2회: loss={second[0]:.4f}, correct={second[1]}, count={second[2]}")
    print("실습 1 확인 통과: 두 번의 업데이트와 기울기 초기화 결과가 맞습니다.")
    ''', tag='exercise-check', exercise='02-step'),
    md('''
    ### 다른 학습률로 시도하기

    아래는 **작은 연습 모델만** 새로 만들어 학습률 `0.01`과 `0.1`로 두 번씩 학습합니다.
    어느 쪽의 두 번째 손실이 더 작을지 예상해 보세요. 본학습 모델이나 정해 둔 학습률은 바꾸지 않습니다.
    '''), code('''
    if not EXERCISE_CHECKS.get("02-step", False):
        raise RuntimeError("실습 1의 확인 셀을 먼저 통과하세요.")
    for practice_lr in (0.01, 0.1):
        trial_model = make_practice_model()
        trial_optimizer = torch.optim.SGD(trial_model.parameters(), lr=practice_lr)
        losses = []
        for _ in range(2):
            result = train_one_batch(trial_model, trial_optimizer, criterion, practice_pixels, practice_labels)
            losses.append(result[0])
        print(f"연습 학습률 {practice_lr}: {losses[0]:.4f} → {losses[1]:.4f}")
    ''', tag='exercise-experiment', exercise='02-step'), md('''
    <details>
    <summary>시도한 뒤 결과 해설 보기</summary>

    backward는 각 가중치를 어느 방향으로 바꿀지 기울기를 계산하고, step은 optimizer가 그 기울기를 사용해
    가중치를 바꾸는 단계입니다. 다음 배치를 학습할 때 zero_grad를 빠뜨리면 이전 기울기가 누적됩니다.
    이 작은 예제에서는 0.1이 더 빠르게 손실을 낮춥니다. 모든 모델에서 학습률이 클수록 좋다는 뜻은 아닙니다.
    아래 실제 이미지 학습은 원래 설정인 Adam과 학습률 0.001을 사용합니다.

    </details>

    ### 같은 함수로 분류 헤드 학습하기

    검증 정확도가 가장 높은 epoch를 선택하고 동률이면 검증 손실이 낮은 모델을 고릅니다.
    테스트 결과로 epoch를 선택하지 않습니다. 이제부터 아래 결과는 실제 GPU 학습 결과입니다.
    ''')]


def scope_exercise():
    return [md('''
    ## 8. 실습 2 · 학습할 가중치 범위 선택하기

    방금 학습한 분류 헤드를 유지한 채, 어느 부분까지 추가로 학습할지 정하는 완성된 함수를 살펴봅니다.

    **제공 입력:** 분류 헤드 학습을 마친 `model`과 문자열 `scope`입니다.
    이번 DeiT에는 `model.vit.encoder.layer`, `model.vit.layernorm`, `model.classifier`가 있습니다.

    **함수의 동작**

    1. `select_finetune_parameters(model, scope)`가 학습할 가중치를 고릅니다.
    2. 호출할 때마다 모든 가중치의 `requires_grad`를 먼저 False로 되돌립니다.
    3. `scope="head"`면 classifier만, `scope="last_block"`이면 마지막 Transformer 블록·최종 layernorm·classifier만 True로 바꿉니다.
    4. 학습할 파라미터만 `{이름: 실제 parameter 객체}` 사전으로 반환합니다.
       이름은 `model.named_parameters()`에서 얻고 새 텐서나 모델을 만들지 않습니다.
    5. 두 범위 이외의 문자열은 ValueError를 내고, 가중치 값 자체는 바꾸지 않습니다.

    **예상 결과:** 현재 모델과 5개 클래스에서는 head만 `965`개, last_block 범위는 `446,213`개입니다.
    모델이나 클래스 수가 달라지면 개수도 달라집니다. 이 숫자를 반환값으로 고정하지 마세요.

    **선택: AI에게 설명 요청하기**

    > 아래 `select_finetune_parameters` 함수에서 head와 last_block의 차이를 설명해 주세요.
    > 범위를 바꿀 때 모든 가중치를 먼저 고정하는 이유와 layer[-1]이 가리키는 위치를 알려 주세요.
    > requires_grad와 파라미터 개수가 무엇을 의미하는지 초보자에게 설명해 주세요.
    > 이 함수가 가중치 값을 바꾸는지, 학습할 대상만 고르는지도 구분해 주세요.
    '''), code('''
    EXERCISE_CHECKS["02-scope"] = False
    FINETUNE_SETUP_COMPLETE = False
    if not HEAD_STAGE_COMPLETE:
        raise RuntimeError("분류 헤드 학습과 평가를 먼저 완료하세요.")
    print("완료한 분류 헤드의 검증 정확도:", f"{head_validation['accuracy']:.1%}")
    print("Transformer 블록 수:", len(model.vit.encoder.layer))
    print("분류할 종류:", model.config.num_labels)
    ''', tag='exercise-input', exercise='02-scope'),
    md('아래 완성된 함수 셀을 실행하세요. 다음 셀에서 범위를 바꾸며 개수와 고정 여부를 확인합니다.'),
    provided_function('02-scope', 'select_finetune_parameters', '''
    def select_finetune_parameters(model, scope):
        if scope not in ("head", "last_block"):
            raise ValueError(f"지원하지 않는 학습 범위: {scope}")
        # 이전에 선택한 범위가 남지 않도록 먼저 모든 가중치를 고정합니다.
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        modules = [model.classifier]
        if scope == "last_block":
            modules.extend([model.vit.encoder.layer[-1], model.vit.layernorm])
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        return {
            name: parameter for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
    '''),
    code('''
    EXERCISE_CHECKS["02-scope"] = False
    if not HEAD_STAGE_COMPLETE:
        raise RuntimeError("분류 헤드 학습을 먼저 완료하세요.")
    if not callable(globals().get("select_finetune_parameters")):
        raise RuntimeError("앞의 select_finetune_parameters 함수 셀을 먼저 실행하세요.")
    unchanged_weights = parameter_sha256(model.named_parameters())
    last_prefix = f"vit.encoder.layer.{len(model.vit.encoder.layer) - 1}."
    parameter_lookup = dict(model.named_parameters())
    for scope in ("head", "last_block", "head", "last_block"):
        chosen = select_finetune_parameters(model, scope)
        prefixes = ("classifier.",) if scope == "head" else (last_prefix, "vit.layernorm.", "classifier.")
        expected = {name for name in parameter_lookup if name.startswith(prefixes)}
        assert isinstance(chosen, dict) and set(chosen) == expected, "선택한 가중치의 이름을 확인하세요."
        assert all(chosen[name] is parameter_lookup[name] for name in expected), "원래 parameter 객체를 반환하세요."
        assert {name for name, p in model.named_parameters() if p.requires_grad} == expected
        assert parameter_sha256(model.named_parameters()) == unchanged_weights, "가중치 값은 바꾸지 마세요."
        print(f"범위 {scope}: {sum(p.numel() for p in chosen.values()):,}개 파라미터")
    try:
        select_finetune_parameters(model, "unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("지원하지 않는 범위에는 ValueError가 필요합니다.")
    # 위 오류 입력까지 확인한 뒤 본학습 범위를 다시 적용합니다.
    chosen = select_finetune_parameters(model, "last_block")
    assert set(chosen) == expected
    assert {n for n, p in model.named_parameters() if p.requires_grad} == expected
    assert all(chosen[name] is parameter_lookup[name] for name in expected)
    assert parameter_sha256(model.named_parameters()) == unchanged_weights
    EXERCISE_CHECKS["02-scope"] = True
    print("실습 2 확인 통과: 범위를 다시 바꿔도 고정 상태와 원래 가중치를 유지했습니다.")
    ''', tag='exercise-check', exercise='02-scope'), md('''
    ### 범위를 바꾼 결과 설명하기

    방금 셀은 head → last_block → head → last_block 순서로 호출했습니다.
    세 번째 출력이 다시 965가 되는 이유와, 마지막에 last_block으로 돌려놓은 이유를 자신의 말로 설명해 보세요.

    <details>
    <summary>시도한 뒤 결과 해설 보기</summary>

    requires_grad는 이번 학습에서 기울기를 계산할 가중치를 고르는 설정입니다.
    범위를 바꿀 때 먼저 전부 고정해야 이전 선택이 남지 않습니다. 학습할 수 있는 가중치가 많아졌다는 사실만으로
    정확도가 높아진다고 판단할 수 없습니다. 아래 실제 학습 뒤 같은 테스트 이미지에서 비교합니다.
    실습의 본학습은 마지막 블록·최종 정규화·분류 헤드로 범위를 맞추고 학습률 0.0001을 사용합니다.

    </details>

    ### 선택한 범위로 실제 파인튜닝하기

    앞의 11개 블록을 고정한 채 방금 학습한 분류 헤드에서 이어서 시작합니다.
    손실 계산과 업데이트에는 실습 1에서 살펴본 함수를 그대로 사용합니다.
    ''')]


def adapt_training(cells):
    assert len(cells) == 28, 'Original training notebook layout changed'
    original = deepcopy(cells)
    assert 'optimizer = torch.optim.Adam(model.classifier.parameters()' in original[15].source
    assert 'last_block_index = ' in original[17].source
    assert 'for epoch in range(1, FINETUNE_EPOCHS' in original[18].source
    original[0].source += '''\n\n**실습 방법:** 핵심 함수 두 개가 완성된 코드로 들어 있습니다. 입력과 함수 동작을 읽고
작은 입력과 범위 변경으로 실행해 본 뒤, 같은 노트북에서 결과와 접힌 해설을 확인하세요.
AI 프롬프트는 코드 설명을 듣거나 조건을 바꿔 볼 때 선택해서 사용합니다.
Codespaces에서 사전 검사를 통과한 뒤 Colab CLI로 실행합니다.
Codespaces CPU의 Run All은 이 GPU 노트북을 실행하지 못합니다. 함수 확인에 실패하면 본학습과 결과 저장을 중단합니다.
완료 보고서가 있는 폴더는 덮어쓰지 않습니다. 본학습을 다시 실행할 때는 기존 결과를 별도로 보관하세요.'''
    result = [original[0], code('''
    TRAINING_ALLOWED = False
    EXERCISE_CHECKS = {"02-step": False, "02-scope": False}
    HEAD_STAGE_COMPLETE = False
    FINETUNE_SETUP_COMPLETE = False
    FINETUNE_STAGE_COMPLETE = False
    MODEL_RELOADED = False
    TABLES_SAVED = False
    REPORT_READY = False
    for name in ("train_one_batch", "select_finetune_parameters", "report"):
        globals().pop(name, None)
    ''', guarded=False, tag='exercise-state'), *original[1:14]]
    result.extend(step_exercise())
    core = '''        optimizer.zero_grad(set_to_none=True)
        logits = model(pixel_values=pixels).logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(labels)
        train_correct += int((logits.argmax(-1) == labels).sum())
        train_count += len(labels)'''
    replacement = '''        loss_value, correct_count, batch_count = train_one_batch(model, optimizer, criterion, pixels, labels)
        train_loss += loss_value * batch_count
        train_correct += correct_count
        train_count += batch_count'''
    head = original[15].source.removeprefix(GUARD)
    assert core in head
    head = head.replace(core, replacement)
    result.append(code(dedent('''
    if not EXERCISE_CHECKS.get("02-step", False):
        raise RuntimeError("실습 1의 확인 셀을 먼저 통과하세요.")
    HEAD_STAGE_COMPLETE = False
    ''').strip() + '\n' + head + '\nHEAD_STAGE_COMPLETE = True'))
    result.extend(scope_exercise())
    scope = original[17].source.removeprefix(GUARD)
    begin = scope.index('for module in (model.vit.encoder.layer[-1]')
    end = scope.index('frozen_before = ')
    scope = scope[:begin] + 'trainable = select_finetune_parameters(model, "last_block")\n' + scope[end:]
    result.append(code(dedent('''
    FINETUNE_SETUP_COMPLETE = False
    if not HEAD_STAGE_COMPLETE or not all(EXERCISE_CHECKS.values()):
        raise RuntimeError("두 실습 확인과 분류 헤드 학습을 먼저 완료하세요.")
    ''').strip() + '\n' + scope + '\nFINETUNE_SETUP_COMPLETE = True'))
    fine = original[18].source.removeprefix(GUARD)
    assert core in fine
    fine = fine.replace(core, replacement)
    result.append(code(dedent('''
    FINETUNE_STAGE_COMPLETE = False
    if not FINETUNE_SETUP_COMPLETE or not all(EXERCISE_CHECKS.values()):
        raise RuntimeError("두 실습 확인과 파인튜닝 준비를 먼저 완료하세요.")
    ''').strip() + '\n' + fine + '\nFINETUNE_STAGE_COMPLETE = True'))
    completed = ('if not FINETUNE_STAGE_COMPLETE:\n'
                 '    raise RuntimeError("실습 확인과 실제 학습을 완료한 뒤 결과를 확인하세요.")\n')
    for index in range(19, 28):
        cell = original[index]
        if cell.cell_type != 'code':
            result.append(cell)
            continue
        body = cell.source.removeprefix(GUARD)
        if index == 23:
            body = 'MODEL_RELOADED = False\n' + body + '\nMODEL_RELOADED = True'
        if index == 25:
            prefix, report = body.split('report = {', 1)
            reload_guard = 'if not MODEL_RELOADED:\n    raise RuntimeError("모델 저장·재로딩 확인을 먼저 완료하세요.")\n'
            result.append(code(completed + reload_guard + 'TABLES_SAVED = False\n' + prefix + '\nTABLES_SAVED = True'))
            result.append(code(completed + reload_guard + 'REPORT_READY = False\nif not TABLES_SAVED:\n    raise RuntimeError("결과 표 저장을 먼저 완료하세요.")\nreport = {' + report + '\nREPORT_READY = True'))
            continue
        if index == 26:
            body = 'if not REPORT_READY:\n    raise RuntimeError("현재 실행의 보고서 준비가 완료되지 않았습니다.")\n' + body
        result.append(code(completed + body))
    for cell in result:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
            assert len(cell.source.splitlines()) <= 40
    return result
