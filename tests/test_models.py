from pathlib import Path

import yaml

from aias_specialist.config import load_settings
from aias_specialist.models import download_baseline_model

ROOT = Path(__file__).resolve().parents[1]


def test_transformers_model_conversion_is_atomic_and_uses_locked_revision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = yaml.safe_load((ROOT / "configs/local_baseline.yaml").read_text(encoding="utf-8"))
    model_dir = tmp_path / "models/converted"
    model_dir.mkdir(parents=True)
    (model_dir / "interrupted.bin").write_bytes(b"partial")
    config["model"].update(
        {
            "repo_id": "openai/whisper-tiny",
            "revision": "main",
            "format": "transformers",
            "conversion_quantization": "int8_float16",
            "local_dir": str(model_dir),
        }
    )
    config["training"] = {"enabled": False}
    config["paths"]["model_lock"] = str(tmp_path / "model-lock.yaml")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    lock = {
        "models": {
            "openai/whisper-tiny": {
                "repo_id": "openai/whisper-tiny",
                "requested_revision": "main",
                "resolved_revision": "fixed-sha",
                "roles": ["benchmark-candidate"],
            }
        }
    }
    Path(config["paths"]["model_lock"]).write_text(
        yaml.safe_dump(lock),
        encoding="utf-8",
    )

    calls: dict[str, object] = {}

    class FakeConverter:
        def __init__(self, repo_id: str, **kwargs) -> None:
            calls["repo_id"] = repo_id
            calls["revision"] = kwargs["revision"]

        def convert(self, output_dir: str, **kwargs) -> str:
            calls["quantization"] = kwargs["quantization"]
            output = Path(output_dir)
            output.mkdir(parents=True)
            for name in ["config.json", "model.bin", "tokenizer.json"]:
                (output / name).write_bytes(b"complete")
            return str(output)

    import ctranslate2.converters

    monkeypatch.setattr(ctranslate2.converters, "TransformersConverter", FakeConverter)

    path, revision = download_baseline_model(load_settings(config_path))

    assert revision == "fixed-sha"
    assert path == model_dir
    assert (model_dir / "model.bin").read_bytes() == b"complete"
    assert calls == {
        "repo_id": "openai/whisper-tiny",
        "revision": "fixed-sha",
        "quantization": "int8_float16",
    }
    preserved = list(model_dir.parent.glob("converted.incomplete-*"))
    assert len(preserved) == 1
    assert (preserved[0] / "interrupted.bin").exists()
