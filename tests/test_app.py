from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "artifacts" / "model.joblib").exists(), reason="artifacts not exported")
def test_streamlit_app_renders_backtest_without_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert at.title[0].value.startswith("🔥")
    assert len(at.tabs) == 3
