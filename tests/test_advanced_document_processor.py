import sys
from types import SimpleNamespace

import pytest

import multimind.document_processing.advanced_document_processor as adp


class FakeDataFrame:
    def __init__(self, data=None, columns=None):
        self.data = list(data) if data is not None else []
        self.columns = list(columns) if columns is not None else []

    def to_string(self):
        return f"columns={self.columns} rows={self.data}"


@pytest.fixture
def processor(monkeypatch):
    monkeypatch.setattr(adp, "PANDAS_AVAILABLE", True)
    monkeypatch.setattr(adp, "pd", SimpleNamespace(DataFrame=FakeDataFrame), raising=False)
    return adp.AdvancedDocumentProcessor(model=object())


@pytest.mark.asyncio
async def test_extract_table_from_rows_first_row_header(processor):
    table = {
        "rows": [["name", "age"], ["alice", 30], ["bob", 25]],
        "confidence": 0.9,
        "position": {"page": 2},
    }
    result = await processor._extract_table_data(table)
    assert result.content.columns == ["name", "age"]
    assert result.content.data == [["alice", 30], ["bob", 25]]
    assert result.confidence == 0.9
    assert result.position == {"page": 2}


@pytest.mark.asyncio
async def test_extract_table_from_rows_explicit_header(processor):
    table = {"rows": [["alice", 30]], "header": ["name", "age"]}
    result = await processor._extract_table_data(table)
    assert result.content.columns == ["name", "age"]
    assert result.content.data == [["alice", 30]]
    assert result.confidence == 1.0


@pytest.mark.asyncio
async def test_extract_table_from_pdf_via_pdfplumber(processor, monkeypatch):
    extracted = [
        [["h1", "h2"], ["a", "b"]],
        [["x1", "x2"], ["c", "d"]],
    ]

    class FakePage:
        def extract_tables(self):
            return extracted

    class FakePDF:
        pages = [FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    opened = []

    def fake_open(path):
        opened.append(path)
        return FakePDF()

    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=fake_open))

    result = await processor._extract_table_data(
        {"pdf_path": "/tmp/doc.pdf", "page": 0, "table_index": 1}
    )
    assert opened == ["/tmp/doc.pdf"]
    assert result.content.columns == ["x1", "x2"]
    assert result.content.data == [["c", "d"]]
    assert result.metadata["tables_on_page"] == 2
    assert result.metadata["page"] == 0


@pytest.mark.asyncio
async def test_extract_table_pdf_page_out_of_range(processor, monkeypatch):
    class FakePDF:
        pages = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=lambda path: FakePDF()))
    with pytest.raises(ValueError, match="out of range"):
        await processor._extract_table_data({"pdf_path": "/tmp/doc.pdf", "page": 3})


@pytest.mark.asyncio
async def test_extract_table_unsupported_shape(processor):
    with pytest.raises(NotImplementedError, match="rows"):
        await processor._extract_table_data({"image": "raw-bytes"})


@pytest.mark.asyncio
async def test_extract_table_requires_pandas(monkeypatch):
    monkeypatch.setattr(adp, "PANDAS_AVAILABLE", False)
    processor = adp.AdvancedDocumentProcessor(model=object())
    with pytest.raises(NotImplementedError, match="pandas"):
        await processor._extract_table_data({"rows": [["a"]]})


@pytest.mark.asyncio
async def test_extract_image_data_still_not_implemented(processor):
    with pytest.raises(NotImplementedError):
        await processor._extract_image_data({"content": b""})
