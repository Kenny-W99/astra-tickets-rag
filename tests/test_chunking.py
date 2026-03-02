from src.utils_text import chunk_text


def test_chunk_text_basic():
    t = "a" * 1000
    chunks = chunk_text(t, chunk_size=400, chunk_overlap=50)
    assert chunks
    assert all(len(c) <= 400 for c in chunks)


def test_chunk_text_overlap_lt_size_required():
    try:
        chunk_text("hello", chunk_size=10, chunk_overlap=10)
        assert False, "expected ValueError"
    except ValueError:
        assert True
