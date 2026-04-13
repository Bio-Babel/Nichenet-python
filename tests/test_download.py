"""Tests for nichenetr._download."""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nichenetr._download import _verify_sha256, resolve_data_path


class TestResolveDataPath:
    def test_local_staging_found(self, tmp_path, monkeypatch):
        """If the file exists in local staging dir, return it directly."""
        staging = tmp_path / "nichenetr_data"
        staging.mkdir()
        f = staging / "test.parquet"
        f.write_text("data")

        monkeypatch.setattr(
            "nichenetr._download._PKG_ROOT",
            tmp_path / "nichenetr_py",
        )
        # _PKG_ROOT.parent / DATA_DIR_NAME => tmp_path / nichenetr_data
        # We need _PKG_ROOT.parent == tmp_path
        monkeypatch.setattr(
            "nichenetr._download._PKG_ROOT",
            tmp_path / "fake_pkg",
        )
        result = resolve_data_path("test.parquet")
        # It should find it in staging since tmp_path/fake_pkg/../nichenetr_data/test.parquet
        assert result == f
        assert result.exists()

    def test_cache_found(self, tmp_path, monkeypatch):
        """If the file exists in cache, return it."""
        # Make local staging not exist
        monkeypatch.setattr(
            "nichenetr._download._PKG_ROOT",
            tmp_path / "no_pkg",
        )
        cache_dir = tmp_path / ".cache" / "nichenetr_py"
        cache_dir.mkdir(parents=True)
        f = cache_dir / "cached.parquet"
        f.write_text("cached_data")

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        result = resolve_data_path("cached.parquet")
        assert result == f

    def test_not_in_registry_raises(self, tmp_path, monkeypatch):
        """If file is not found locally and not in registry, raise."""
        monkeypatch.setattr(
            "nichenetr._download._PKG_ROOT",
            tmp_path / "no_pkg",
        )
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.setattr("nichenetr._download.REGISTRY", {})

        with pytest.raises(FileNotFoundError, match="not found locally"):
            resolve_data_path("nonexistent.parquet")

    def test_no_url_in_registry_raises(self, tmp_path, monkeypatch):
        """If registry entry has no URL, raise."""
        monkeypatch.setattr(
            "nichenetr._download._PKG_ROOT",
            tmp_path / "no_pkg",
        )
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.setattr(
            "nichenetr._download.REGISTRY",
            {"nourl.parquet": {"sha256": "abc"}},
        )

        with pytest.raises(FileNotFoundError, match="no download URL"):
            resolve_data_path("nourl.parquet")

    def test_download_and_verify(self, tmp_path, monkeypatch):
        """If file needs downloading, download and verify sha256."""
        monkeypatch.setattr(
            "nichenetr._download._PKG_ROOT",
            tmp_path / "no_pkg",
        )
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        content = b"hello world"
        sha = hashlib.sha256(content).hexdigest()

        monkeypatch.setattr(
            "nichenetr._download.REGISTRY",
            {"dl.parquet": {"url": "http://fake/dl.parquet", "sha256": sha}},
        )

        def mock_download(url, dest):
            dest.write_bytes(content)

        monkeypatch.setattr("nichenetr._download._download", mock_download)

        result = resolve_data_path("dl.parquet")
        assert result.exists()
        assert result.read_bytes() == content


class TestDownloadFunction:
    def test_download_with_content_length(self, tmp_path):
        """Test the actual _download function with a mocked urlopen."""
        from nichenetr._download import _download
        from unittest.mock import MagicMock, patch
        import io

        content = b"hello world data " * 100
        mock_resp = MagicMock()
        mock_resp.headers.get.return_value = str(len(content))
        mock_resp.read = MagicMock(side_effect=[content, b""])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            dest = tmp_path / "downloaded.bin"
            _download("http://fake/url", dest)
        assert dest.exists()
        assert dest.read_bytes() == content

    def test_download_without_content_length(self, tmp_path):
        """Test download when Content-Length is 0 (unknown size)."""
        from nichenetr._download import _download
        from unittest.mock import MagicMock, patch

        content = b"small data"
        mock_resp = MagicMock()
        mock_resp.headers.get.return_value = "0"
        mock_resp.read = MagicMock(side_effect=[content, b""])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            dest = tmp_path / "dl2.bin"
            _download("http://fake/url", dest)
        assert dest.read_bytes() == content


class TestVerifySha256:
    def test_correct_hash_passes(self, tmp_path):
        f = tmp_path / "good.bin"
        content = b"test data"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        # Should not raise
        _verify_sha256(f, expected)

    def test_wrong_hash_raises_and_deletes(self, tmp_path):
        f = tmp_path / "bad.bin"
        f.write_bytes(b"actual data")
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            _verify_sha256(f, "0000000000000000deadbeef")
        assert not f.exists()

    def test_none_hash_skips(self, tmp_path):
        f = tmp_path / "any.bin"
        f.write_bytes(b"whatever")
        # Should not raise
        _verify_sha256(f, None)
