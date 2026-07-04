"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { FileText, RefreshCw, Wrench } from "lucide-react";

import AppShell from "../../components/AppShell";
import DefectHeatmap from "../../components/DefectHeatmap";
import InspectionResult from "../../components/InspectionResult";
import ProductionMetadataForm, { EMPTY_CATALOG, EMPTY_METADATA } from "../../components/ProductionMetadataForm";
import SeverityBadge from "../../components/SeverityBadge";
import { formatDateTime } from "../../services/dateTime";
import { getCurrentUser } from "../../services/authApi";
import { formatReviewStatus } from "../../services/inspectionLabels";
import { listInspections, updateInspectionMetadata, updateReviewStatus } from "../../services/inspectionApi";
import { getProductionCatalog } from "../../services/productionApi";
import { createInspectionReport } from "../../services/reportApi";
import { getReworkTicketByInspection } from "../../services/reworkApi";

const REVIEW_ROLES = new Set(["admin", "quality_manager", "factory_supervisor"]);

export default function InspectionPage() {
  const [inspections, setInspections] = useState([]);
  const [selected, setSelected] = useState(null);
  const [message, setMessage] = useState("");
  const [filters, setFilters] = useState({ productId: "", productionLine: "", reviewStatus: "" });
  const [reviewNotes, setReviewNotes] = useState("");
  const [currentUser, setCurrentUser] = useState(null);
  const [reworkTicket, setReworkTicket] = useState(null);
  const [catalog, setCatalog] = useState(EMPTY_CATALOG);
  const [metadataForm, setMetadataForm] = useState(EMPTY_METADATA);

  async function loadInspections() {
    const payload = await listInspections({ limit: 50, ...filters });
    setInspections(payload.items || []);
    setSelected((current) => current || payload.items?.[0] || null);
  }

  useEffect(() => {
    loadInspections().catch(() => setMessage("Could not load inspections"));
    getProductionCatalog()
      .then(setCatalog)
      .catch(() => setCatalog(EMPTY_CATALOG));
    getCurrentUser()
      .then(setCurrentUser)
      .catch(() => setCurrentUser(null));
  }, []);

  useEffect(() => {
    setReviewNotes(selected?.review_notes || "");
    setReworkTicket(null);
    setMetadataForm({
      ...EMPTY_METADATA,
      batch_number: selected?.batch_number || "",
      product_id: selected?.product_id || "",
      production_line: selected?.production_line || "",
      shift: selected?.shift || "",
      operator_name: selected?.operator_name || "",
      source_label: selected?.source_label || "",
    });
    if (selected?.review_status === "sent_for_rework" && selected?.id) {
      getReworkTicketByInspection(selected.id)
        .then(setReworkTicket)
        .catch(() => setReworkTicket(null));
    }
  }, [selected?.id]);

  async function handleReviewStatus(status) {
    if (!selected?.id) return;
    const updated = await updateReviewStatus(selected.id, status, reviewNotes);
    setSelected(updated);
    setInspections((items) => items.map((item) => (item.id === updated.id ? updated : item)));
    if (status === "sent_for_rework") {
      setReworkTicket({
        id: updated.rework_ticket_id,
        ticket_number: updated.rework_ticket_number,
        status: updated.rework_ticket_status,
      });
      setMessage(`Rework ticket created: ${updated.rework_ticket_number || updated.rework_ticket_id}`);
    } else {
      setReworkTicket(null);
      setMessage(`Review status updated to ${formatReviewStatus(status)}`);
    }
    setReviewNotes("");
  }

  async function handleMetadataSave() {
    if (!selected?.id) return;
    const updated = await updateInspectionMetadata(selected.id, metadataForm);
    setSelected(updated);
    setInspections((items) => items.map((item) => (item.id === updated.id ? updated : item)));
    setMessage("Production metadata updated");
  }

  async function handleReport() {
    if (!selected?.id) return;
    const report = await createInspectionReport(selected.id);
    setMessage(`Report generated: ${report.id}`);
  }

  return (
    <AppShell title="Inspection History" subtitle="Review stored inspection results and quality decisions.">
      <div className="page-actions">
        <button className="ghost-button" type="button" onClick={loadInspections}>
          <RefreshCw size={16} />
          Refresh
        </button>
        <button className="ghost-button" type="button" onClick={handleReport} disabled={!selected}>
          <FileText size={16} />
          Generate report
        </button>
        {message ? <span className="inline-success">{message}</span> : null}
      </div>

      <section className="tool-panel">
        <div className="metadata-grid">
          <label>
            Product ID
            <select
              value={filters.productId}
              onChange={(event) => setFilters((current) => ({ ...current, productId: event.target.value }))}
            >
              <option value="">All products</option>
              {catalog.products.map((product) => (
                <option key={product.product_id} value={product.product_id}>
                  {product.product_id}
                </option>
              ))}
            </select>
          </label>
          <label>
            Production line
            <select
              value={filters.productionLine}
              onChange={(event) => setFilters((current) => ({ ...current, productionLine: event.target.value }))}
            >
              <option value="">All lines</option>
              {catalog.production_lines.map((line) => (
                <option key={line.line_id} value={line.line_id}>
                  {line.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Review status
            <select
              value={filters.reviewStatus}
              onChange={(event) => setFilters((current) => ({ ...current, reviewStatus: event.target.value }))}
            >
              <option value="">All statuses</option>
              <option value="ai_completed">AI completed</option>
              <option value="manual_review">Manual review</option>
              <option value="approved">Approved</option>
              <option value="rejected">Rejected</option>
              <option value="sent_for_rework">Sent for rework</option>
            </select>
          </label>
          <button className="primary-button" type="button" onClick={loadInspections}>
            Apply filters
          </button>
        </div>
      </section>

      <div className="history-layout">
        <section className="tool-panel">
          <div className="panel-heading">
            <div>
              <h2>Records</h2>
              <p>{inspections.length} inspections</p>
            </div>
          </div>
          <div className="record-list">
            {inspections.map((item) => (
              <button
                key={item.id}
                className={selected?.id === item.id ? "record-item active" : "record-item"}
                type="button"
                onClick={() => setSelected(item)}
              >
                <span>
                  <strong>{item.prediction || "Pending"}</strong>
                  <small>
                    {item.product_id || "No product"} | {item.defect_type || "Unknown"} |{" "}
                    {formatDateTime(item.created_at)}
                  </small>
                </span>
                <SeverityBadge level={item.severity_level} />
              </button>
            ))}
            {!inspections.length ? <div className="empty-visual">No inspection records found.</div> : null}
          </div>
        </section>

        <div className="stack inspection-detail-stack">
          <InspectionResult result={selected} />
          <section className="tool-panel">
            <div className="panel-heading">
              <div>
                <h2>Manual Review</h2>
                <p>{formatReviewStatus(selected?.review_status) || "No inspection selected"}</p>
              </div>
            </div>
            <label className="notes-field">
              Reviewer notes
              <textarea
                value={reviewNotes}
                onChange={(event) => setReviewNotes(event.target.value)}
                placeholder={selected?.review_notes || "Add approval, rejection, or rework notes"}
              />
            </label>
            {reworkTicket?.id ? (
              <div className="rework-callout">
                <span>
                  <Wrench size={16} />
                  <strong>{reworkTicket.ticket_number || "Rework ticket"}</strong>
                  <small>{reworkTicket.status ? formatReviewStatus(reworkTicket.status) : "Open"}</small>
                </span>
                <Link className="primary-link" href="/rework">
                  Open queue
                </Link>
              </div>
            ) : null}
            {REVIEW_ROLES.has(currentUser?.role) ? (
              <div className="page-actions compact review-actions">
                <button
                  className="ghost-button"
                  type="button"
                  onClick={() => handleReviewStatus("approved")}
                  disabled={!selected}
                >
                  Approve
                </button>
                <button
                  className="ghost-button"
                  type="button"
                  onClick={() => handleReviewStatus("rejected")}
                  disabled={!selected}
                >
                  Reject
                </button>
                <button
                  className="ghost-button"
                  type="button"
                  onClick={() => handleReviewStatus("sent_for_rework")}
                  disabled={!selected}
                >
                  Rework
                </button>
              </div>
            ) : (
              <p className="form-success">Review actions are available to supervisors and quality managers.</p>
            )}
          </section>
          <section className="tool-panel">
            <div className="panel-heading">
              <div>
                <h2>Production Metadata</h2>
                <p>Assign traceability details for audits and reports.</p>
              </div>
            </div>
            <ProductionMetadataForm
              value={metadataForm}
              catalog={catalog}
              onChange={setMetadataForm}
              disabled={!selected}
              placeholders={{ batch: "Unassigned", product: "Unassigned", line: "Unassigned" }}
            />
            <div className="page-actions compact">
              <button className="primary-button" type="button" onClick={handleMetadataSave} disabled={!selected}>
                Save metadata
              </button>
            </div>
          </section>
          <DefectHeatmap imageUrl={selected?.heatmap_url} />
        </div>
      </div>
    </AppShell>
  );
}
