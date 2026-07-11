from ask_santosh.textutils import clean_html


def test_strips_tags_and_collapses_whitespace():
    assert clean_html("<p>hello   <b>world</b></p>\n\n") == "hello world"


def test_unescapes_entities():
    assert clean_html("a &amp; b &lt;c&gt;") == "a & b <c>"


def test_handles_empty():
    assert clean_html("") == ""
