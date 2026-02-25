"""Smoke test to verify the test harness works."""


def test_import_errors_module() -> None:
    from src.infra.errors import DistRLError, WorkerFailureError

    err = WorkerFailureError(worker_id="w-1", reason="simulated crash")
    assert isinstance(err, DistRLError)
    assert "w-1" in str(err)
