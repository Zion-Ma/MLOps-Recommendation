from collections import Counter
from pathlib import Path

import pytest
import yaml

K8S_DIR = Path(__file__).resolve().parents[1] / "k8s"

MANIFEST_FILES = sorted(K8S_DIR.rglob("*.yaml"))

EXPECTED_KINDS = {
    "Namespace",
    "ConfigMap",
    "Secret",
    "PersistentVolumeClaim",
    "Deployment",
    "StatefulSet",
    "Service",
    "HorizontalPodAutoscaler",
    "Job",
    "ServiceAccount",
    "Role",
    "RoleBinding",
}


def load_documents():
    documents = []
    for path in MANIFEST_FILES:
        for document in yaml.safe_load_all(path.read_text()):
            if document is not None:
                documents.append((path.name, document))
    return documents


DOCUMENTS = load_documents()


def test_manifest_files_exist():
    assert len(MANIFEST_FILES) >= 10


DOCUMENT_IDS = [
    f"{name}:{doc.get('kind')}/{doc.get('metadata', {}).get('name')}"
    for name, doc in DOCUMENTS
]


@pytest.mark.parametrize("filename,document", DOCUMENTS, ids=DOCUMENT_IDS)
def test_document_has_required_fields(filename, document):
    assert "apiVersion" in document
    assert document.get("kind") in EXPECTED_KINDS
    assert document.get("metadata", {}).get("name")
    if document["kind"] != "Namespace":
        assert document["metadata"].get("namespace") == "news-rec"


def test_all_expected_components_present():
    kinds = Counter(document["kind"] for _, document in DOCUMENTS)
    names = {
        (document["kind"], document["metadata"]["name"]) for _, document in DOCUMENTS
    }

    for component in [
        "api",
        "postgres",
        "redis",
        "opensearch",
        "qdrant",
        "mlflow",
        "airflow",
        "prometheus",
        "grafana",
    ]:
        assert any(name.startswith(component) for kind, name in names), component

    assert ("HorizontalPodAutoscaler", "api") in names
    assert kinds["PersistentVolumeClaim"] >= 2
    assert kinds["Job"] >= 2


def test_deployments_and_statefulsets_have_probes_and_resources():
    for filename, document in DOCUMENTS:
        if document["kind"] not in ("Deployment", "StatefulSet"):
            continue
        pod_spec = document["spec"]["template"]["spec"]
        for container in pod_spec["containers"]:
            label = f"{filename}:{container['name']}"
            assert "resources" in container, label
            # The scheduler has no HTTP endpoint; every other long-running
            # container must declare a readiness probe.
            if container["name"] != "scheduler":
                assert "readinessProbe" in container, label


def test_selectors_match_pod_labels():
    for filename, document in DOCUMENTS:
        if document["kind"] not in ("Deployment", "StatefulSet"):
            continue
        selector = document["spec"]["selector"]["matchLabels"]
        labels = document["spec"]["template"]["metadata"]["labels"]
        assert selector.items() <= labels.items(), filename


def test_services_target_existing_pod_labels():
    pod_labels = []
    for _, document in DOCUMENTS:
        if document["kind"] in ("Deployment", "StatefulSet"):
            pod_labels.append(document["spec"]["template"]["metadata"]["labels"])

    for filename, document in DOCUMENTS:
        if document["kind"] != "Service":
            continue
        selector = document["spec"]["selector"]
        assert any(
            selector.items() <= labels.items() for labels in pod_labels
        ), f"{filename}: no pods match service selector {selector}"
