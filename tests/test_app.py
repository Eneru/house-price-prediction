from streamlit.testing.v1 import AppTest


def test_incorrect_field_typing():
    at = AppTest.from_file("app.py", default_timeout=360)
    at.session_state["filters"] = [{"field": "year_built", "value": "not a valid year"}]
    at.run()
    at.button("predict_button").click().run()
    assert len(at.error) == 1
    assert "Invalid value for year_built" in at.error[0].value


def test_correct_field_typing():
    at = AppTest.from_file("app.py", default_timeout=360)
    at.session_state["filters"] = [{"field": "year_built", "value": "2008"}]
    at.run()
    at.button("predict_button").click().run()
    assert len(at.error) == 0
    assert len(at.success) == 1
    assert "💰 Estimated Sale Price: " in at.success[0].value


def test_field_filling():
    at = AppTest.from_file("app.py", default_timeout=360)
    at.run()
    at.selectbox("field_0").select("year_built").run()
    at.text_input("value_0").input("2008").run()
    assert len(at.error) == 0
    assert len(at.success) == 0
    assert len(at.selectbox) == 1
    assert len(at.text_input) == 1
    assert at.session_state["filters"][0]["field"] == "year_built"
    assert at.session_state["filters"][0]["value"] == "2008"


def test_field_adding():
    at = AppTest.from_file("app.py", default_timeout=360)
    at.run()
    at.selectbox("field_0").select("year_built").run()
    at.text_input("value_0").input("2008").run()
    assert len(at.selectbox) == 1
    assert len(at.text_input) == 1
    at.button("add_filter_button").click().run()
    at.selectbox("field_1").select("garage_cars").run()
    at.text_input("value_1").input("3").run()
    assert len(at.error) == 0
    assert len(at.success) == 0
    assert len(at.selectbox) == 2
    assert len(at.text_input) == 2
    assert at.session_state["filters"][0]["field"] == "year_built"
    assert at.session_state["filters"][0]["value"] == "2008"
    assert at.session_state["filters"][1]["field"] == "garage_cars"
    assert at.session_state["filters"][1]["value"] == "3"


def test_field_removal():
    at = AppTest.from_file("app.py", default_timeout=360)
    at.session_state["filters"] = [
        {"field": "year_built", "value": "2008"},
        {"field": "garage_cars", "value": "2.0"},
    ]
    at.run()
    assert len(at.error) == 0
    assert len(at.success) == 0
    assert len(at.selectbox) == 2
    assert len(at.text_input) == 2
    at.button("remove_1").click().run()
    assert len(at.error) == 0
    assert len(at.success) == 0
    assert len(at.selectbox) == 1
    assert len(at.text_input) == 1
    assert at.session_state["filters"][0]["field"] == "year_built"
    assert at.session_state["filters"][0]["value"] == "2008"
