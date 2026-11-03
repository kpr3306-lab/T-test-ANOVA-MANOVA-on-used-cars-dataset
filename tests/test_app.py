"""Smoke test: every page renders without exceptions, and body-type inference works."""
from streamlit.testing.v1 import AppTest

from utils.preprocessing import infer_body_type

PAGES = ["Overview", "Market Explorer", "Dataset", "T-Test", "ANOVA", "MANOVA"]


def test_body_type():
    assert infer_body_type("Hyundai Creta 1.6 CRDi SX", 5) == "SUV"
    assert infer_body_type("Honda City 1.5 V MT", 5) == "Sedan"
    assert infer_body_type("Toyota Etios Liva G", 5) == "Hatchback"
    assert infer_body_type("Toyota Innova 2.5 G", 7) == "MUV"
    assert infer_body_type("Unknown Car X", 7) == "MUV"


def test_pages_render():
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    assert not at.exception
    radio = at.sidebar.radio[0]
    for opt in radio.options:
        radio.set_value(opt).run()
        assert not at.exception, (opt, at.exception)
    assert [o.split(": ", 1)[1] for o in radio.options] == PAGES
