from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py", default_timeout=900)
at.run()
assert not at.exception, at.exception

# shrink generation for test speed: sliders in sidebar order
# 0 gamma, 1 delta, 2 z_threshold, 3 temperature, 4 top_p, 5 max_new
at.sidebar.slider[5].set_value(80)
at.run()

buttons = {b.label: b for b in at.button}
buttons["Generate both"].click()
at.run()
assert not at.exception, at.exception
assert "marked" in at.session_state and len(at.session_state["marked"]) > 50
zs = [m.value for m in at.metric if "z" not in dir(m)] if False else [m.value for m in at.metric]
print("metrics rendered:", [(m.label, m.value) for m in at.metric])
assert any(m.label == "z-score" for m in at.metric)

buttons = {b.label: b for b in at.button}
buttons["Detect"].click()
at.run()
assert not at.exception, at.exception
html_blocks = [el.value for el in at.markdown if "border-bottom" in str(el.value)]
assert html_blocks, "token highlight html missing"
print("token highlight rendered, length:", len(html_blocks[-1]))

buttons = {b.label: b for b in at.button}
buttons["Run robustness sweeps"].click()
at.run()
assert not at.exception, at.exception
print("dataframes rendered:", len(at.dataframe))
assert len(at.dataframe) == 2
print("ALL APP TESTS PASSED")
