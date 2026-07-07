import sys
from types import SimpleNamespace

import pytest
pytest.importorskip("numpy")  # requires optional extras absent on core-only installs

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


@pytest.fixture
def ocr(monkeypatch):
    calls = []

    def image_to_string(pixels):
        calls.append(pixels)
        return "ocr text"

    monkeypatch.setattr(adp, "PYTESSERACT_AVAILABLE", True)
    monkeypatch.setattr(
        adp, "pytesseract", SimpleNamespace(image_to_string=image_to_string), raising=False
    )
    return calls


@pytest.mark.asyncio
async def test_extract_image_from_array(processor, ocr):
    import numpy as np

    pixels = np.zeros((2, 2), dtype=np.uint8)
    result = await processor._extract_image_data({"array": pixels, "metadata": {"page": 1}})
    assert result.text == "ocr text"
    assert result.content is pixels
    assert ocr == [pixels]
    assert result.objects == []
    assert result.captions == []
    assert result.metadata["source"] == "array"
    assert result.metadata["extractors"] == ["pytesseract"]
    assert result.metadata["page"] == 1
    assert result.metadata["objects_extracted"] is False


@pytest.mark.asyncio
async def test_extract_image_from_path_via_pil(processor, ocr, monkeypatch):
    import numpy as np

    pixels = np.ones((3, 3), dtype=np.uint8)

    class FakePILImage:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __array__(self, dtype=None, copy=None):
            return pixels

    opened = []

    def fake_open(path):
        opened.append(path)
        return FakePILImage()

    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=SimpleNamespace(open=fake_open)))

    result = await processor._extract_image_data({"path": "/tmp/scan.png"})
    assert opened == ["/tmp/scan.png"]
    assert result.text == "ocr text"
    assert (result.content == pixels).all()
    assert result.metadata["source"] == "/tmp/scan.png"
    assert result.metadata["extractors"] == ["PIL", "pytesseract"]


@pytest.mark.asyncio
async def test_extract_image_requires_pytesseract(processor, monkeypatch):
    monkeypatch.setattr(adp, "PYTESSERACT_AVAILABLE", False)
    with pytest.raises(NotImplementedError, match="pytesseract"):
        await processor._extract_image_data({"array": object()})


@pytest.mark.asyncio
async def test_extract_image_unsupported_shape(processor, ocr):
    with pytest.raises(NotImplementedError, match="Supported shapes"):
        await processor._extract_image_data({"content": b""})
