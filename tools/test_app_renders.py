"""Render every app page through Streamlit's test harness and fail on any exception."""

from streamlit.testing.v1 import AppTest

PAGES = ["Portfolio", "Data & leakage", "Models", "Patterns", "Simulator"]


def check(page: str) -> None:
    at = AppTest.from_file("app.py", default_timeout=180).run()
    if at.exception:
        raise AssertionError(f"{page}: initial run raised {at.exception[0].value}")
    at.sidebar.radio[0].set_value(page).run()
    if at.exception:
        raise AssertionError(f"{page}: {at.exception[0].value}")
    print(f"OK  {page}")


def sweep_cutoff() -> None:
    at = AppTest.from_file("app.py", default_timeout=180).run()
    at.sidebar.radio[0].set_value("Portfolio").run()
    for value in (0.10, 0.50, 0.90):
        slider = next(w for w in at.slider if "scoring" in w.label.lower())
        slider.set_value(value).run()
        if at.exception:
            raise AssertionError(f"cutoff {value}: {at.exception[0].value}")
        print(f"OK  Portfolio @ cutoff {value}")


def simulator_value() -> None:
    at = AppTest.from_file("app.py", default_timeout=180).run()
    at.sidebar.radio[0].set_value("Simulator").run()
    if at.exception:
        raise AssertionError(f"Simulator: {at.exception[0].value}")
    text = " ".join(element.value for element in at.markdown)
    if "1.4%" not in text:
        raise AssertionError("Simulator did not score the default applicant at 1.4%")
    print("OK  Simulator scores the default applicant at 1.4%")


for name in PAGES:
    check(name)
sweep_cutoff()
simulator_value()
print("all pages render clean")
