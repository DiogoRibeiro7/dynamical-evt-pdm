import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from dyn_evt_pdm.data import acquisition


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.position = 0

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            result = self.payload[self.position :]
            self.position = len(self.payload)
            return result
        result = self.payload[self.position : self.position + size]
        self.position += len(result)
        return result


def test_planned_files_resolves_public_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(url: object, timeout: int) -> FakeResponse:
        del timeout
        text = str(url)
        if "6854240" in text:
            return FakeResponse(
                json.dumps(
                    {
                        "files": [
                            {
                                "key": "dataset_train.csv",
                                "size": 10,
                                "checksum": "md5:abc",
                                "links": {"self": "https://example.test/metropt"},
                            }
                        ]
                    }
                ).encode()
            )
        if "7766691" in text:
            return FakeResponse(
                json.dumps(
                    {
                        "files": [
                            {
                                "key": "MetroPT2.csv",
                                "size": 20,
                                "checksum": "md5:def",
                                "links": {"self": "https://example.test/metropt2"},
                            },
                            {
                                "key": "dataset_train.csv",
                                "size": 10,
                                "checksum": "md5:abc",
                                "links": {"self": "https://example.test/duplicate"},
                            },
                        ]
                    }
                ).encode()
            )
        html = (
            '<a href="https://api.researchdata.se/dataset/2024-34/3/file/data?'
            'filePath=train_tte.csv">download</a>'
        )
        return FakeResponse(html.encode())

    monkeypatch.setattr(acquisition.urllib.request, "urlopen", fake_urlopen)

    files = acquisition.planned_files(["all"], raw_root=tmp_path)

    assert [file.filename for file in files] == [
        "dataset_train.csv",
        "MetroPT2.csv",
        "train_tte.csv",
        "condition+monitoring+of+hydraulic+systems.zip",
        "secom.zip",
    ]
    assert files[0].destination == tmp_path / "metropt" / "dataset_train.csv"
    assert files[1].destination == tmp_path / "metropt2" / "MetroPT2.csv"
    assert files[2].destination == tmp_path / "scania_component_x" / "train_tte.csv"
    assert files[3].destination == (
        tmp_path / "hydraulic_systems" / "condition+monitoring+of+hydraulic+systems.zip"
    )
    assert files[4].destination == tmp_path / "secom" / "secom.zip"


def test_fetch_datasets_downloads_validates_and_reuses_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"timestamp,value\n2022-01-01,1\n"
    checksum = hashlib.md5(payload, usedforsecurity=False).hexdigest()

    def fake_planned_files(
        dataset_names: list[str], *, raw_root: Path
    ) -> list[acquisition.RemoteFile]:
        assert dataset_names == ["metropt"]
        return [
            acquisition.RemoteFile(
                dataset="metropt",
                filename="dataset_train.csv",
                url="https://example.test/dataset_train.csv",
                destination=raw_root / "metropt" / "dataset_train.csv",
                size_bytes=len(payload),
                checksum=f"md5:{checksum}",
            )
        ]

    def fake_urlopen(request: Any, timeout: int) -> FakeResponse:
        del request, timeout
        return FakeResponse(payload)

    monkeypatch.setattr(acquisition, "planned_files", fake_planned_files)
    monkeypatch.setattr(acquisition.urllib.request, "urlopen", fake_urlopen)

    first = acquisition.fetch_datasets(["metropt"], raw_root=tmp_path)
    second = acquisition.fetch_datasets(["metropt"], raw_root=tmp_path)

    assert first[0].status == "downloaded"
    assert second[0].status == "exists"
    assert (tmp_path / "metropt" / "dataset_train.csv").read_bytes() == payload
    manifest = json.loads((tmp_path / "metropt" / "manifest.json").read_text())
    assert manifest[0]["filename"] == "dataset_train.csv"
    assert manifest[0]["checksum"] == f"md5:{checksum}"


def test_planned_files_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        acquisition.planned_files(["not-a-dataset"])
