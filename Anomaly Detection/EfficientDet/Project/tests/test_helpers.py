"""Keep test fixtures in an explicit scratch folder, separate from results."""
import os
import uuid
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def test_directory():
    root = Path(os.environ.get("VEHICLE_TEST_SCRATCH", ".cache/test-fixtures")).resolve()
    target = (root / uuid.uuid4().hex).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Invalid test scratch path")
    target.mkdir(parents=True)
    # Preserve fixtures for debugging; they are ignored by Git and never exported.
    yield target
