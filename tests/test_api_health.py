from app import api


def test_liveness_does_not_depend_on_model_promotion():
    assert api.health() == {
        "status": "ok",
        "service": "banking-risk-ml-platform",
    }


def test_readiness_degrades_when_no_production_model(monkeypatch):
    def missing_production(_stage: str):
        raise KeyError("no production model")

    monkeypatch.setattr(api.REGISTRY, "latest", missing_production)
    result = api.readiness()

    assert result["status"] == "degraded"
    assert result["ready"] is False
    assert result["reason"] == "no production model"


def test_readiness_degrades_when_artifact_verification_fails(monkeypatch):
    class Record:
        version = "risk-test"

    monkeypatch.setattr(api.REGISTRY, "latest", lambda _stage: Record())
    monkeypatch.setattr(api.REGISTRY, "verify_artifact", lambda _version: False)

    result = api.readiness()
    assert result["status"] == "degraded"
    assert result["ready"] is False
    assert result["artifact_verified"] is False
