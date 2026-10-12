import { useState } from "react";
import { ApiError, EvidenceFormat, getEvidenceReport } from "../api/client";
import { toApiError } from "../api/useApi";
import BackendDown from "../components/BackendDown";

export default function EvidencePage() {
  const [format, setFormat] = useState<EvidenceFormat>("md");
  const [downloading, setDownloading] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const [previewHtml, setPreviewHtml] = useState<string | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  const download = async () => {
    setDownloading(true);
    setError(null);
    try {
      const { blob, filename } = await getEvidenceReport(format);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(toApiError(err));
    } finally {
      setDownloading(false);
    }
  };

  const preview = async () => {
    setPreviewing(true);
    setError(null);
    try {
      const { blob } = await getEvidenceReport("html");
      setPreviewHtml(await blob.text());
    } catch (err) {
      setError(toApiError(err));
    } finally {
      setPreviewing(false);
    }
  };

  return (
    <>
      <h1 className="page-header">Evidence Report</h1>
      <p className="page-sub">
        Export a point-in-time governance evidence report for auditors and reviews.
      </p>

      <div className="card">
        <div className="form-row">
          <div>
            <label htmlFor="evidence-format">Format</label>
            <select
              id="evidence-format"
              value={format}
              onChange={(e) => setFormat(e.target.value as EvidenceFormat)}
            >
              <option value="md">Markdown (.md)</option>
              <option value="html">HTML (.html)</option>
            </select>
          </div>
          <button onClick={download} disabled={downloading}>
            {downloading ? "Preparing..." : "Download report"}
          </button>
          <button className="secondary" onClick={preview} disabled={previewing}>
            {previewing ? "Loading preview..." : "Preview inline"}
          </button>
        </div>
      </div>

      {error ? <BackendDown backend="dashboard" error={error} /> : null}

      {previewHtml != null ? (
        <div className="card">
          <h2>Preview</h2>
          <iframe
            className="evidence-frame"
            title="Evidence report preview"
            sandbox=""
            srcDoc={previewHtml}
          />
        </div>
      ) : null}
    </>
  );
}
