"""Exercise shell orchestration and cleanup without Google auth or a GPU allocation."""
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / 'hf_colab_gpu/run_pipeline_colab.sh'

FAKE_CLI = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
root = Path(os.environ['FAKE_COLAB_STATE'])
mode = os.environ.get('FAKE_COLAB_MODE', 'success')
args = sys.argv[1:]
assert args[:2] == ['--auth', 'adc']
args = args[2:]
command = args[0]
with (root / 'commands.jsonl').open('a') as stream:
    stream.write(json.dumps(args) + '\n')
state_path = root / 'state.json'
state = json.loads(state_path.read_text())
if command == 'sessions':
    if not state['sessions']:
        print('[colab] No active sessions found on server.')
    for name, endpoint in state['sessions'].items():
        print(f'[{name}] {endpoint} | Hardware: T4 | Variant: GPU')
elif command == 'new':
    assert args[-2:] == ['--gpu', 'T4']
    state['sessions']['?' if mode == 'unidentified_creation' else args[args.index('-s') + 1]] = 'owned-endpoint'
    state_path.write_text(json.dumps(state))
    if mode in ('create_error_after_creation', 'unidentified_creation'):
        raise SystemExit(2)
elif command == 'upload':
    source, remote = args[-2:]
    (root / Path(remote).name).write_bytes(Path(source).read_bytes())
elif command == 'exec':
    # The real CLI may return zero even when remote Python raises an exception.
    assert '-f' in args and '--timeout' in args
    if Path(args[args.index('-f') + 1]).name == 'install.py':
        source = Path(args[args.index('-f') + 1]).read_text()
        assert "'pip', 'install'" in source and "check=True" in source
        assert "'pip', 'check'" not in source
    print('remote call ended')
elif command == 'restart-kernel':
    pass
elif command == 'download':
    remote, destination = args[-2:]
    destination = Path(destination)
    config = json.loads((root / 'run-config.json').read_text())
    if remote.endswith('install-complete.json'):
        if mode == 'missing_install':
            raise SystemExit(0)
        report = {'status': 'installed', 'run_id': config['run_id'], 'files': config['files'],
                  'versions': {'torch': '2.8.0+cu126', 'torchvision': '0.23.0+cu126',
                               'transformers': '4.57.6', 'huggingface-hub': '0.36.0'}}
    else:
        state['downloads'] = state.get('downloads', 0) + 1
        state_path.write_text(json.dumps(state))
        if mode == 'retry_download' and state['downloads'] == 1:
            raise SystemExit(1)
        if mode == 'missing_result':
            raise SystemExit(0)
        report = {k: config[k] for k in ('run_id', 'model_id', 'model_revision', 'image_sha256')}
        report.update(status='completed', task='image-classification', device='cuda', device_name='Tesla T4',
                      predictions=[{'label': str(index), 'score': value}
                                   for index, value in enumerate([0.5, 0.2, 0.1, 0.09, 0.06])],
                      versions={'torch': '2.8.0+cu126', 'transformers': '4.57.6', 'huggingface-hub': '0.36.0'})
        if mode == 'malformed_result':
            report['predictions'][0]['score'] = float('nan')
        elif mode == 'wrong_run_id':
            report['run_id'] = 'earlier-run'
    destination.write_text(json.dumps(report))
elif command == 'stop':
    name = args[args.index('-s') + 1]
    if mode == 'stop_failure':
        raise SystemExit(3)
    state['sessions'].pop(name)
    state_path.write_text(json.dumps(state))
else:
    raise SystemExit('unexpected fake command: ' + command)
'''


@pytest.fixture
def rig(tmp_path):
    fake = tmp_path / 'colab'
    fake.write_text(FAKE_CLI)
    fake.chmod(0o700)
    state = tmp_path / 'state'
    state.mkdir()
    (state / 'state.json').write_text(json.dumps({'sessions': {'unrelated-work': 'other-endpoint'}}))
    image = tmp_path / 'original.png'
    Image.new('L', (7, 5), 90).save(image)
    output = tmp_path / 'output'
    env = {**os.environ, 'PYTHON_BIN': sys.executable, 'COLAB_BIN': str(fake),
           'FAKE_COLAB_STATE': str(state)}

    def run(mode='success', *, session='our-demo', extra=()):
        return subprocess.run(
            ['bash', str(WRAPPER), '--auth', 'adc', '--image', str(image),
             '--output-dir', str(output), '--session', session, *extra],
            env={**env, 'FAKE_COLAB_MODE': mode}, capture_output=True, text=True, timeout=30)

    def commands():
        return [json.loads(line) for line in (state / 'commands.jsonl').read_text().splitlines()]

    return run, output, state, commands


def assert_only_own_session_stopped(state, commands):
    assert [args for args in commands() if args[0] == 'stop'] == [['stop', '-s', 'our-demo']]
    assert json.loads((state / 'state.json').read_text())['sessions'] == {'unrelated-work': 'other-endpoint'}


def test_success_requires_result_and_confirmed_cleanup(rig):
    run, output, state, commands = rig
    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert '시연 완료:' in result.stdout
    assert_only_own_session_stopped(state, commands)
    report = json.loads((output / 'result.json').read_text())
    cleanup = json.loads((output / 'cleanup.json').read_text())
    assert cleanup == {'run_id': report['run_id'], 'session': 'our-demo',
                       'termination_status': 'confirmed', 'result_verified': True, 'success': True}
    assert Image.open(output / 'input.png').mode == 'RGB'
    assert (output.stat().st_mode & 0o777) == 0o700
    assert [args[0] for args in commands()].count('new') == 1


@pytest.mark.parametrize('mode', ['missing_install', 'missing_result', 'malformed_result', 'wrong_run_id'])
def test_remote_zero_exit_is_not_success_without_valid_evidence(rig, mode):
    run, output, state, commands = rig
    result = run(mode)
    assert result.returncode != 0
    assert '시연 완료:' not in result.stdout
    assert_only_own_session_stopped(state, commands)
    cleanup = json.loads((output / 'cleanup.json').read_text())
    assert cleanup['termination_status'] == 'confirmed' and not cleanup['success']
    assert not cleanup['result_verified']


def test_existing_name_never_creates_or_stops_a_session(rig):
    run, output, state, commands = rig
    result = run(session='unrelated-work')
    assert result.returncode != 0
    assert [args[0] for args in commands()] == ['sessions']
    assert json.loads((output / 'cleanup.json').read_text())['termination_status'] == 'not_started'
    assert json.loads((state / 'state.json').read_text())['sessions'] == {'unrelated-work': 'other-endpoint'}


def test_creation_error_after_server_allocation_still_cleans_up(rig):
    run, output, state, commands = rig
    result = run('create_error_after_creation')
    assert result.returncode != 0
    assert_only_own_session_stopped(state, commands)
    assert json.loads((output / 'cleanup.json').read_text())['termination_status'] == 'confirmed'


def test_failed_stop_prevents_a_success_claim(rig):
    run, output, state, commands = rig
    result = run('stop_failure')
    assert result.returncode != 0 and '시연 완료:' not in result.stdout
    cleanup = json.loads((output / 'cleanup.json').read_text())
    assert cleanup['result_verified'] and not cleanup['success']
    assert cleanup['termination_status'] == 'unconfirmed'
    assert [args for args in commands() if args[0] == 'stop'] == [['stop', '-s', 'our-demo']]


def test_result_download_retries_once_without_a_second_allocation(rig):
    run, output, state, commands = rig
    result = run('retry_download')
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((state / 'state.json').read_text())['downloads'] == 2
    assert [args[0] for args in commands()].count('new') == 1
    assert_only_own_session_stopped(state, commands)


def test_existing_output_is_preserved_without_remote_calls(rig):
    run, output, state, commands = rig
    output.mkdir()
    original = output / 'keep.txt'
    original.write_text('existing result')
    result = run()
    assert result.returncode != 0 and original.read_text() == 'existing result'
    assert not (state / 'commands.jsonl').exists()


def test_dry_run_needs_neither_cli_nor_input_and_creates_nothing(tmp_path):
    output = tmp_path / 'must-not-exist'
    result = subprocess.run(
        ['bash', str(WRAPPER), '--dry-run', '--image', 'missing-image.png', '--output-dir', str(output)],
        env={**os.environ, 'COLAB_BIN': '/does/not/exist', 'PYTHON_BIN': '/does/not/exist'},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and '실제로 실행하지 않음' in result.stdout
    assert not output.exists()


def test_unidentifiable_creation_never_stops_other_or_orphan_sessions(rig):
    run, output, state, commands = rig
    result = run('unidentified_creation')
    assert result.returncode != 0 and '시연 완료:' not in result.stdout
    assert not any(args[0] == 'stop' for args in commands())
    cleanup = json.loads((output / 'cleanup.json').read_text())
    assert cleanup['termination_status'] == 'unconfirmed' and not cleanup['success']
    assert json.loads((state / 'state.json').read_text())['sessions']['unrelated-work'] == 'other-endpoint'


def test_broken_symlink_output_never_allocates(rig):
    run, output, state, commands = rig
    output.symlink_to(output.parent / 'nonexistent-destination', target_is_directory=True)
    result = run()
    assert result.returncode != 0 and output.is_symlink()
    assert not (state / 'commands.jsonl').exists()
