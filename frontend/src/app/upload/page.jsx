"use client";

import { useEffect, useState } from "react";
import { FileText } from "lucide-react";

import AppShell from "../../components/AppShell";
import BatchInspectionResults from "../../components/BatchInspectionResults";
import DefectHeatmap from "../../components/DefectHeatmap";
import ImageUpload from "../../components/ImageUpload";
import InspectionResult from "../../components/InspectionResult";
import ProductionMetadataForm, { EMPTY_CATALOG, EMPTY_METADATA } from "../../components/ProductionMetadataForm";
import { createInspectionReport } from "../../services/reportApi";
import { inspectBatch, inspectImage } from "../../services/inspectionApi";
import { getProductionCatalog } from "../../services/productionApi";

export default function UploadPage() {
  const [file, setFile] = useState(null);
  const [metadata, setMetadata] = useState(EMPTY_METADATA);
  const [batchFiles, setBatchFiles] = useState([]);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [batchResults, setBatchResults] = useState([]);
  const [batchSummary, setBatchSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [batchLoading, setBatchLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [failure, setFailure] = useState(null);
  const [catalog, setCatalog] = useState(EMPTY_CATALOG);

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }
    const nextPreviewUrl = URL.createObjectURL(file);
    setPreviewUrl(nextPreviewUrl);
    return () => URL.revokeObjectURL(nextPreviewUrl);
  }, [file]);

  useEffect(() => {
    getProductionCatalog()
      .then(setCatalog)
      .catch(() => setCatalog(EMPTY_CATALOG));
  }, []);

  async function handleInspect() {
    if (!file) return;
    setLoading(true);
    setMessage("");
    setFailure(null);
    try {
      const inspection = await inspectImage(file, metadata);
      setResult(inspection);
      setMessage("Inspection completed");
    } catch (err) {
      setFailure({
        title: "Inspection failed",
        message: err.message || "The image could not be inspected.",
        status: err.status,
        requestId: err.requestId,
      });
    } finally {
      setLoading(false);
    }
  }

  async function handleBatchInspect() {
    if (!batchFiles.length) return;
    setBatchLoading(true);
    setMessage("");
    setFailure(null);
    try {
      const payload = await inspectBatch(batchFiles, metadata);
      setBatchResults(payload.items || []);
      setBatchSummary(payload.summary || null);
      setResult(payload.items?.[0] || null);
      setMessage(`Batch complete: ${payload.total} images inspected`);
    } catch (err) {
      setFailure({
        title: "Batch inspection failed",
        message: err.message || "The selected images could not be inspected.",
        status: err.status,
        requestId: err.requestId,
      });
    } finally {
      setBatchLoading(false);
    }
  }

  async function handleReport() {
    if (!result?.id) return;
    setFailure(null);
    try {
      const report = await createInspectionReport(result.id);
      setMessage(`Report generated: ${report.id}`);
    } catch (err) {
      setFailure({
        title: "Report generation failed",
        message: err.message || "The report could not be generated.",
        status: err.status,
        requestId: err.requestId,
      });
    }
  }

  return (
    <AppShell title="Image Inspection" subtitle="Upload a product image and run AI defect detection.">
      <div className="inspection-layout">
        <ImageUpload file={file} onFileChange={setFile} onInspect={handleInspect} loading={loading} />
        <InspectionResult result={result} />
      </div>

      <section className="tool-panel">
        <div className="panel-heading">
          <div>
            <h2>Production Metadata</h2>
            <p>Attach manufacturing context to this inspection.</p>
          </div>
        </div>
        <ProductionMetadataForm value={metadata} catalog={catalog} onChange={setMetadata} />
      </section>

      <div className="inspection-layout">
        <section className="tool-panel">
          <div className="panel-heading">
            <div>
              <h2>Original Image</h2>
              <p>Input image preview.</p>
            </div>
          </div>
          {previewUrl ? (
            <div className="image-frame">
              <img src={previewUrl} alt="Selected product" />
            </div>
          ) : (
            <div className="empty-visual">Preview appears after file selection.</div>
          )}
        </section>
        <DefectHeatmap imageUrl={result?.heatmap_url} />
      </div>

      <div className="page-actions">
        <button className="ghost-button" type="button" onClick={handleReport} disabled={!result?.id}>
          <FileText size={16} />
          Generate PDF report
        </button>
        {message ? <span className="inline-success">{message}</span> : null}
      </div>
      {failure ? (
        <section className="error-panel">
          <strong>{failure.title}</strong>
          <p>{failure.message}</p>
          <small>
            {failure.status ? `Status ${failure.status}` : "Request failed"}
            {failure.requestId ? ` | Request ID ${failure.requestId}` : ""}
          </small>
        </section>
      ) : null}

      <section className="tool-panel">
        <div className="panel-heading">
          <div>
            <h2>Batch Image Processing</h2>
            <p>Inspect multiple production images in one request.</p>
          </div>
        </div>

        <div className="batch-controls">
          <label className="file-picker">
            <input
              type="file"
              accept="image/png,image/jpeg,image/jpg,image/bmp,image/tiff,image/webp"
              multiple
              onChange={(event) => setBatchFiles(Array.from(event.target.files || []).slice(0, 20))}
            />
            <span>{batchFiles.length ? `${batchFiles.length} files selected` : "Select batch images"}</span>
          </label>
          <button
            className="primary-button"
            type="button"
            onClick={handleBatchInspect}
            disabled={!batchFiles.length || batchLoading}
          >
            {batchLoading ? "Processing" : "Run Batch"}
          </button>
        </div>

        <BatchInspectionResults results={batchResults} summary={batchSummary} />
      </section>
    </AppShell>
  );
}
