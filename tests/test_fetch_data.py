import pytest

from scripts.fetch_data import fetch_kudago_data


def test_fetch_data():
    try:
        data = fetch_kudago_data("places/", params={"page_size": 1})
        assert isinstance(data, dict), "API response is not a dictionary"
        assert "results" in data, "API response missing 'results' key"
        assert len(data["results"]) > 0, "No data returned from API"
    except Exception as e:
        pytest.fail(f"API request failed: {str(e)}")
