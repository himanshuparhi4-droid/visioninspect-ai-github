"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, FileText, ScanSearch } from "lucide-react";

import AppShell from "../../components/AppShell";
import BatchInspectionResults from "../../components/BatchInspectionResults";
import DefectHeatmap from "../../components/DefectHeatmap";
import ImageUpload from "../../components/ImageUpload";
import InspectionResult from "../../components/InspectionResult";
import ProductionMetadataForm, { EMPTY_CATALOG, EMPTY_METADATA } from "../../components/ProductionMetadataForm";
import { getCurrentUser } from "../../services/authApi";
import { createInspectionReport } from "../../services/reportApi";
import { getModelCategories, inspectBatch, inspectImage } from "../../services/inspectionApi";
import { getProductionCatalog } from "../../services/productionApi";

function automaticMetadata(file, catalog, current) {
  if (!file) return current;
  const category = current.category || "";

  // Try to find an active batch matching the category if one is set
  let batch = null;
  if (category && catalog.batches) {
    const categoryProductIds = new Set(
      (catalog.products || []).filter((product) => product.category === category).map((product) => product.product_id)
    );
    batch =
      catalog.batches.find((item) => item.status === "active" && categoryProductIds.has(item.product_id)) ||
      catalog.batches.find((item) => categoryProductIds.has(item.product_id));
  }

  const stem = file.name
    .replace(/\.[^.]+$/, "")
    .replace(/[^a-zA-Z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  const date = new Date().toISOString().slice(0, 10).replaceAll("-", "");

  // Find a product that matches the category
  let defaultProduct = null;
  if (category && catalog.products) {
    defaultProduct = catalog.products.find((p) => p.category === category)?.product_id;
  }

  return {
    ...current,
    batch_number: current.batch_number || batch?.batch_number || `AUTO-${date}`,
    product_id:
      current.product_id ||
      batch?.product_id ||
      defaultProduct ||
      `${(category || "ITEM").toUpperCase()}-${stem.toUpperCase()}`,
    production_line:
      current.production_line || batch?.production_line || catalog.production_lines?.[0]?.line_id || "Line-Manual-01",
    shift: current.shift || batch?.shift || catalog.shifts?.[0] || "Auto Shift",
    source_label: current.source_label || file.name,
  };
}

export default function UploadPage() {
  const [file, setFile] = useState(null);
  const [metadata, setMetadata] = useState(EMPTY_METADATA);
  const [batchFiles, setBatchFiles] = useState([]);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [batchResults, setBatchResults] = useState([]);
  const [batchSummary, setBatchSummary] = useState(null);
  const [batchFailures, setBatchFailures] = useState([]);
  const [loading, setLoading] = useState(false);
  const [batchLoading, setBatchLoading] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [failure, setFailure] = useState(null);
  const [slowMessage, setSlowMessage] = useState("");
  const [catalog, setCatalog] = useState(EMPTY_CATALOG);
  const [modelCategories, setModelCategories] = useState([]);

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
    setMetadata((current) => automaticMetadata(file, catalog, current));
  }, [file, catalog]);

  useEffect(() => {
    getProductionCatalog()
      .then(setCatalog)
      .catch(() => setCatalog(EMPTY_CATALOG));
    getModelCategories()
      .then((payload) => {
        const categories = payload.items || [];
        setModelCategories(categories);
      })
      .catch(() => setModelCategories([]));
    getCurrentUser()
      .then((user) => setMetadata((current) => ({ ...current, operator_name: current.operator_name || user.name })))
      .catch(() => undefined);
  }, []);

  function handleFileChange(nextFile) {
    setFile(nextFile);
    setResult(null);
    setFailure(null);
    setMessage("");
    if (nextFile) {
      setMetadata((current) => ({ ...automaticMetadata(nextFile, catalog, current), source_label: nextFile.name }));
    }
  }

  function handleMetadataChange(nextMetadata) {
    const categoryChanged = nextMetadata.category !== metadata.category;
    setMetadata(categoryChanged && file ? automaticMetadata(file, catalog, nextMetadata) : nextMetadata);
    setResult(null);
    setFailure(null);
    setMessage("");
  }

  function handleBatchFiles(nextFiles) {
    const selected = Array.from(nextFiles || []).slice(0, 20);
    setBatchFiles(selected);
    if (selected[0]) {
      setMetadata((current) => ({ ...automaticMetadata(selected[0], catalog, current), source_label: "" }));
    }
  }

  async function handleInspect() {
    if (!file) return;
    if (!metadata.category) {
      setFailure({
        title: "Inspection category required",
        message: "Choose the product category before running the inspection.",
        status: 400,
      });
      return;
    }
    setLoading(true);
    setMessage("");
    setFailure(null);
    setResult(null);
    setSlowMessage("");
    const slowTimer = window.setTimeout(
      () => setSlowMessage("Inspection is taking longer than expected. The request is still processing."),
      20000
    );
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
      window.clearTimeout(slowTimer);
      setSlowMessage("");
      setLoading(false);
    }
  }

  async function handleBatchInspect() {
    if (!batchFiles.length) return;
    if (!metadata.category) {
      setFailure({
        title: "Inspection category required",
        message: "Choose one product category before running the batch. Every image in the batch must belong to it.",
      });
      return;
    }
    setBatchLoading(true);
    setMessage("");
    setFailure(null);
    setSlowMessage("");
    setBatchResults([]);
    setBatchSummary(null);
    setBatchFailures([]);
    const slowTimer = window.setTimeout(
      () => setSlowMessage("Batch inspection is taking longer than expected. The request is still processing."),
      20000
    );
    try {
      const payload = await inspectBatch(batchFiles, metadata);
      setBatchResults(payload.items || []);
      setBatchSummary(payload.summary || null);
      setBatchFailures(payload.failures || []);
      setResult(payload.items?.[0] || null);
      setMessage(
        `Batch complete: ${payload.summary?.succeeded ?? payload.total} inspected, ${payload.summary?.failed ?? 0} failed`
      );
    } catch (err) {
      setBatchResults([]);
      setBatchSummary(null);
      setBatchFailures([]);
      setFailure({
        title: "Batch inspection failed",
        message: err.message || "The selected images could not be inspected.",
        status: err.status,
        requestId: err.requestId,
      });
    } finally {
      window.clearTimeout(slowTimer);
      setSlowMessage("");
      setBatchLoading(false);
    }
  }

  async function handleReport() {
    if (!result?.id || reportLoading) return;
    setFailure(null);
    setReportLoading(true);
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
    } finally {
      setReportLoading(false);
    }
  }

  return (
    <AppShell title="Image Inspection" subtitle="Upload a product image and run AI defect detection.">
      <section className="workflow-banner" aria-label="Inspection workflow">
        {[
          ["Upload", file ? "Image selected" : "Choose image"],
          ["Metadata", metadata.product_id ? "Context added" : "Add details"],
          ["Inspect", loading ? "Running AI" : result ? "Completed" : "Run model"],
          ["Review", result ? "Ready" : "View result"],
        ].map(([step, description], index) => {
          const active =
            (index === 0 && file) ||
            (index === 1 && metadata.product_id) ||
            (index === 2 && loading) ||
            (index === 3 && result);
          return (
            <div key={step} className={active ? "workflow-step active" : "workflow-step"}>
              <strong>{index + 1}</strong>
              <span>
                <b>{step}</b>
                <small>{description}</small>
              </span>
            </div>
          );
        })}
      </section>

      {failure ? (
        <section className="error-panel" role="alert">
          <strong>{failure.title}</strong>
          <p>{failure.message}</p>
          <small>
            {failure.status ? `Status ${failure.status}` : "Request failed"}
            {failure.requestId ? ` | Request ID ${failure.requestId}` : ""}
          </small>
        </section>
      ) : null}

      {slowMessage ? (
        <section className="processing-warning" role="status">
          <strong>Inspection delayed</strong>
          <p>{slowMessage}</p>
        </section>
      ) : null}

      {file && !metadata.category ? (
        <section className="processing-warning category-selection-prompt" role="status">
          <strong>Choose the inspection category</strong>
          <p>Select the product type below so VisionInspect AI uses the correct detector and defect classifier.</p>
        </section>
      ) : null}

      <div className="workflow-grid">
        <div className="stack">
          <ImageUpload
            file={file}
            onFileChange={handleFileChange}
            onInspect={handleInspect}
            loading={loading || batchLoading}
          />
          <section className="tool-panel">
            <div className="panel-heading">
              <div>
                <h2>Production Metadata</h2>
                <p>Attach manufacturing context before inspection.</p>
              </div>
              <CheckCircle2 size={22} />
            </div>
            <ProductionMetadataForm
              value={metadata}
              catalog={catalog}
              modelCategories={modelCategories}
              onChange={handleMetadataChange}
              disabled={loading || batchLoading}
            />
          </section>
        </div>

        <div className="stack">
          <InspectionResult result={result} />
          <section className="tool-panel run-summary-panel">
            <div className="panel-heading">
              <div>
                <h2>Inspection Actions</h2>
                <p>Generate reports after a successful result.</p>
              </div>
              <ScanSearch size={22} />
            </div>
            <div className="page-actions compact">
              <button
                className="ghost-button"
                type="button"
                onClick={handleReport}
                disabled={!result?.id || reportLoading}
              >
                <FileText size={16} />
                {reportLoading ? "Generating report" : "Generate PDF report"}
              </button>
              {message ? <span className="inline-success">{message}</span> : null}
            </div>
          </section>
        </div>
      </div>

      <div className="media-grid">
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

      <section className="tool-panel batch-panel">
        <div className="panel-heading">
          <div>
            <h2>Batch Image Processing</h2>
            <p>
              {metadata.category
                ? `Using the ${metadata.category.replaceAll("_", " ")} inspection model for every selected image.`
                : "Choose one inspection category above before selecting a same-category batch."}
            </p>
          </div>
        </div>

        <div className="batch-controls">
          <label className="file-picker">
            <input
              type="file"
              accept="image/png,image/jpeg,image/jpg,image/bmp,image/tiff,image/webp"
              multiple
              onChange={(event) => handleBatchFiles(event.target.files)}
            />
            <span>{batchFiles.length ? `${batchFiles.length} files selected` : "Select batch images"}</span>
          </label>
          <button
            className="primary-button"
            type="button"
            onClick={handleBatchInspect}
            disabled={!batchFiles.length || !metadata.category || batchLoading || loading}
          >
            {batchLoading ? "Processing" : "Run Batch"}
          </button>
        </div>

        <BatchInspectionResults results={batchResults} summary={batchSummary} failures={batchFailures} />
      </section>
    </AppShell>
  );
}
