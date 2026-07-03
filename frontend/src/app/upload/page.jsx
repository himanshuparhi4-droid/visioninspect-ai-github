"use client";

import { useEffect, useState } from "react";
import { FileText } from "lucide-react";

import AppShell from "../../components/AppShell";
import DefectHeatmap from "../../components/DefectHeatmap";
import ImageUpload from "../../components/ImageUpload";
import InspectionResult from "../../components/InspectionResult";
import { createInspectionReport } from "../../services/reportApi";
import { inspectBatch, inspectImage } from "../../services/inspectionApi";
import { getProductionCatalog } from "../../services/productionApi";

export default function UploadPage() {
  const [file, setFile] = useState(null);
  const [metadata, setMetadata] = useState({
    batch_number: "",
    product_id: "",
    production_line: "",
    shift: "",
    operator_name: "",
    source_label: "",
  });
  const [batchFiles, setBatchFiles] = useState([]);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [batchResults, setBatchResults] = useState([]);
  const [batchSummary, setBatchSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [batchLoading, setBatchLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [catalog, setCatalog] = useState({ products: [], production_lines: [], batches: [], shifts: [] });

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
    getProductionCatalog().then(setCatalog).catch(() => setCatalog({ products: [], production_lines: [], batches: [], shifts: [] }));
  }, []);

  async function handleInspect() {
    if (!file) return;
    setLoading(true);
    setMessage("");
    try {
      const inspection = await inspectImage(file, metadata);
      setResult(inspection);
    } catch (err) {
      setMessage(err.message || "Inspection failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleBatchInspect() {
    if (!batchFiles.length) return;
    setBatchLoading(true);
    setMessage("");
    try {
      const payload = await inspectBatch(batchFiles, metadata);
      setBatchResults(payload.items || []);
      setBatchSummary(payload.summary || null);
      setResult(payload.items?.[0] || null);
      setMessage(`Batch complete: ${payload.total} images inspected`);
    } catch (err) {
      setMessage(err.message || "Batch inspection failed");
    } finally {
      setBatchLoading(false);
    }
  }

  function updateMetadata(field, value) {
    setMetadata((current) => ({ ...current, [field]: value }));
  }

  function selectBatch(batchNumber) {
    const batch = catalog.batches.find((item) => item.batch_number === batchNumber);
    setMetadata((current) => ({
      ...current,
      batch_number: batchNumber,
      product_id: batch?.product_id || current.product_id,
      production_line: batch?.production_line || current.production_line,
      shift: batch?.shift || current.shift,
    }));
  }

  async function handleReport() {
    if (!result?.id) return;
    const report = await createInspectionReport(result.id);
    setMessage(`Report generated: ${report.id}`);
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
        <div className="metadata-grid">
          <label>
            Batch number
            <select value={metadata.batch_number} onChange={(event) => selectBatch(event.target.value)}>
              <option value="">Select batch</option>
              {catalog.batches.map((batch) => (
                <option key={batch.batch_number} value={batch.batch_number}>{batch.batch_number}</option>
              ))}
            </select>
          </label>
          <label>
            Product ID
            <select value={metadata.product_id} onChange={(event) => updateMetadata("product_id", event.target.value)}>
              <option value="">Select product</option>
              {catalog.products.map((product) => (
                <option key={product.product_id} value={product.product_id}>{product.product_id} - {product.name}</option>
              ))}
            </select>
          </label>
          <label>
            Production line
            <select value={metadata.production_line} onChange={(event) => updateMetadata("production_line", event.target.value)}>
              <option value="">Select line</option>
              {catalog.production_lines.map((line) => (
                <option key={line.line_id} value={line.line_id}>{line.name}</option>
              ))}
            </select>
          </label>
          <label>
            Shift
            <select value={metadata.shift} onChange={(event) => updateMetadata("shift", event.target.value)}>
              <option value="">Unassigned</option>
              {catalog.shifts.map((shift) => (
                <option key={shift} value={shift}>{shift}</option>
              ))}
            </select>
          </label>
          <label>
            Operator
            <input value={metadata.operator_name} onChange={(event) => updateMetadata("operator_name", event.target.value)} />
          </label>
          <label>
            Source label
            <input value={metadata.source_label} onChange={(event) => updateMetadata("source_label", event.target.value)} />
          </label>
        </div>
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
        {message ? <span className={message.includes("failed") ? "inline-error" : "inline-success"}>{message}</span> : null}
      </div>

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
          <button className="primary-button" type="button" onClick={handleBatchInspect} disabled={!batchFiles.length || batchLoading}>
            {batchLoading ? "Processing" : "Run Batch"}
          </button>
        </div>

        {batchResults.length ? (
          <>
          <div className="stats-grid compact-stats">
            <div className="stat-card"><small>Total</small><strong>{batchSummary?.total ?? batchResults.length}</strong></div>
            <div className="stat-card"><small>Good</small><strong>{batchSummary?.good ?? batchResults.filter((item) => item.prediction === "Good").length}</strong></div>
            <div className="stat-card"><small>Defective</small><strong>{batchSummary?.defective ?? batchResults.filter((item) => item.prediction === "Defective").length}</strong></div>
            <div className="stat-card"><small>Failed</small><strong>{batchSummary?.fail ?? batchResults.filter((item) => item.pass_fail === "Fail").length}</strong></div>
            <div className="stat-card"><small>Critical</small><strong>{batchSummary?.critical ?? batchResults.filter((item) => item.severity_level === "Critical").length}</strong></div>
            <div className="stat-card"><small>Avg Confidence</small><strong>{batchSummary?.average_confidence != null ? `${(batchSummary.average_confidence * 100).toFixed(1)}%` : "Pending"}</strong></div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Prediction</th>
                  <th>Defect</th>
                  <th>Severity</th>
                  <th>Decision</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {batchResults.map((item) => (
                  <tr key={item.id}>
                    <td>{item.prediction}</td>
                    <td>{item.defect_type}</td>
                    <td>{item.severity_level}</td>
                    <td>{item.pass_fail}</td>
                    <td>{item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : "Pending"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          </>
        ) : null}
      </section>
    </AppShell>
  );
}
