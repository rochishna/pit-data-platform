"""Downloader tests with a fake API: pagination, missing values, and that the key never leaks."""
import urllib.parse

import pytest

from pit import download

KEY = "abcd1234secretkey"


def fake_api(observations, page_size=2, error=None):
    """Simulates the FRED endpoint, paging `observations` by offset."""
    calls = []

    def fetch(url):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        calls.append(q)
        if error:
            raise error
        off = int(q["offset"][0])
        return {"count": len(observations), "observations": observations[off:off + page_size]}
    return fetch, calls


def obs(date, value, start, end="9999-12-31"):
    return {"date": date, "value": value, "realtime_start": start, "realtime_end": end}


def test_paginates_and_parses(monkeypatch):
    rows = [obs("2024-01-01", "100.0", "2024-02-15", "2024-03-14"),
            obs("2024-01-01", "100.4", "2024-03-15"),
            obs("2024-02-01", ".", "2024-03-15"),
            obs("2024-02-01", "101.2", "2024-04-16")]
    fetch, calls = fake_api(rows, page_size=3)
    df = download.fetch_vintages("INDPRO", KEY, fetch=fetch)
    assert len(calls) == 2                                   # 3 + 1 rows
    assert df.height == 4
    assert df["value"].to_list() == [100.0, 100.4, None, 101.2]   # '.' -> null
    assert str(df["realtime_end"][1]) == "9999-12-31"
    assert calls[0]["realtime_start"] == ["1776-07-04"]       # asks for ALL vintages


def test_key_is_redacted_from_errors():
    err = RuntimeError(f"HTTP 500 for https://api.stlouisfed.org/...&api_key={KEY}&offset=0")
    fetch, _ = fake_api([], error=err)
    with pytest.raises(download.DownloadError) as exc:
        download.fetch_vintages("INDPRO", KEY, fetch=fetch)
    assert KEY not in str(exc.value)
    assert "***" in str(exc.value)


def test_api_error_message_is_redacted():
    def fetch(url):
        return {"error_message": f"Bad Request. The value for variable api_key={KEY} is not registered."}
    with pytest.raises(download.DownloadError) as exc:
        download.fetch_vintages("INDPRO", KEY, fetch=fetch)
    assert KEY not in str(exc.value)


def test_short_download_is_an_error():
    def fetch(url):
        return {"count": 5, "observations": []}
    with pytest.raises(download.DownloadError, match="empty page"):
        download.fetch_vintages("INDPRO", KEY, fetch=fetch)


def test_key_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text(f'FRED_API_KEY="{KEY}"\n')
    assert download.load_api_key(env) == KEY


def test_missing_key_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(download.DownloadError, match="FRED_API_KEY"):
        download.load_api_key(tmp_path / "nope.env")
