"""Document loader module for loading and ingesting documents.

Requires the ``documents`` extras (``pdfplumber``, ``python-docx``, ``pillow``, …):
``pip install 'multimind-sdk[documents]'``.
"""

try:
    from .data_ingestion import DataIngestion
    from .document_loader import (
        AudioDocumentLoader,
        BaseDocumentLoader,
        DatabaseDocumentLoader,
        DefaultFileLoader,
        DocumentConnector,
        DocumentFormat,
        DocumentLoaderFactory,
        DocumentMetadata,
        DocumentSource,
        EmailDocumentLoader,
        ImageDocumentLoader,
        LoadedDocument,
        LocalDocumentLoader,
        PresentationDocumentLoader,
        SpreadsheetDocumentLoader,
        StreamDocumentLoader,
        VideoDocumentLoader,
        WebDocumentLoader,
        WebsiteDocumentLoader,
    )
except ImportError as exc:  # pragma: no cover - exercised on minimal installs
    raise ImportError(
        "Document loading features require additional dependencies. "
        "Install with: pip install 'multimind-sdk[documents]'"
    ) from exc

__all__ = [
    "DataIngestion",
    "DocumentMetadata",
    "LoadedDocument",
    "DocumentFormat",
    "DocumentSource",
    "DocumentConnector",
    "BaseDocumentLoader",
    "LocalDocumentLoader",
    "WebDocumentLoader",
    "DatabaseDocumentLoader",
    "StreamDocumentLoader",
    "DocumentLoaderFactory",
    "WebsiteDocumentLoader",
    "EmailDocumentLoader",
    "SpreadsheetDocumentLoader",
    "PresentationDocumentLoader",
    "ImageDocumentLoader",
    "AudioDocumentLoader",
    "VideoDocumentLoader",
    "DefaultFileLoader",
]
