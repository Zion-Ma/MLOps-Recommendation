import json

from pipelines.alert_if_degraded import check_ctr, check_drift, check_latency, run


def write_json(path, payload):
    path.write_text(json.dumps(payload))
    return path


def test_check_ctr_flags_low_ctr():
    assert check_ctr({"ctr": 0.001}, min_ctr=0.01)
    assert check_ctr({"ctr": 0.05}, min_ctr=0.01) == []
    assert check_ctr({"ctr": None}, min_ctr=0.01) == []
    assert check_ctr(None, min_ctr=0.01) == []


def test_check_latency_flags_latency_and_errors():
    report = {"latency_seconds": 2.0, "error_rate": 0.2}

    violations = check_latency(report, max_latency_seconds=1.0, max_error_rate=0.05)

    assert len(violations) == 2


def test_check_drift_flags_high_drift_share():
    report = {"drift_share": 0.8, "drifted_features": ["bm25_score"]}

    assert check_drift(report, max_drift_share=0.5)
    assert check_drift({"drift_share": 0.1}, max_drift_share=0.5) == []


def test_run_healthy_system(tmp_path):
    ctr_path = write_json(tmp_path / "ctr.json", {"ctr": 0.1})
    output_path = tmp_path / "alerts.json"

    report = run(output_path, ctr_path=ctr_path)

    assert report["degraded"] is False
    assert json.loads(output_path.read_text())["violations"] == []


def test_run_degraded_system(tmp_path):
    ctr_path = write_json(tmp_path / "ctr.json", {"ctr": 0.0001})
    latency_path = write_json(tmp_path / "latency.json", {"latency_seconds": 5.0})
    output_path = tmp_path / "alerts.json"

    report = run(
        output_path,
        ctr_path=ctr_path,
        latency_path=latency_path,
        min_ctr=0.01,
        max_latency_seconds=1.0,
    )

    assert report["degraded"] is True
    assert len(report["violations"]) == 2
