from multimind.splitter import TextSplitter

def test_text_splitter_split_text():
    splitter = TextSplitter(chunk_size=5, overlap=2)
    chunks = splitter.split_text("abcdefghij")
    assert all(isinstance(c, str) for c in chunks)
    assert len(chunks) > 0

def test_text_splitter_split_by_sentences():
    splitter = TextSplitter()
    sentences = splitter.split_by_sentences("Hello world! How are you? I am fine.")
    assert sentences == ["Hello world", "How are you", "I am fine"]

def test_text_splitter_split_by_paragraphs():
    splitter = TextSplitter()
    paragraphs = splitter.split_by_paragraphs("Para1\n\nPara2\n\nPara3")
    assert paragraphs == ["Para1", "Para2", "Para3"] 