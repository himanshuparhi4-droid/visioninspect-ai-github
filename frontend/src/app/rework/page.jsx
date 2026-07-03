"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, RefreshCw, Save, Wrench } from "lucide-react";

import AppShell from "../../components/AppShell";
import SeverityBadge from "../../components/SeverityBadge";
import { formatDateTime } from "../../services/dateTime";
import { listReworkTickets, updateReworkTicket } from "../../services/reworkApi";

const statusOptions = ["open", "in_progress", "completed", "closed"];
const priorityOptions = ["Low", "Medium", "High", "Critical"];

function statusLabel(value) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function draftFromTicket(ticket) {
  return {
    assigned_to: ticket.assigned_to || "",
    priority: ticket.priority || "Medium",
    status: ticket.status || "open",
    reason: ticket.reason || "",
    resolution_notes: ticket.resolution_notes || "",
  };
}

export default function ReworkPage() {
  const [tickets, setTickets] = useState([]);
  const [drafts, setDrafts] = useState({});
  const [status, setStatus] = useState("");
  const [message, setMessage] = useState("");

  async function loadTickets() {
    setMessage("");
    try {
      const items = await listReworkTickets(status);
      setTickets(items);
      setDrafts(Object.fromEntries(items.map((ticket) => [ticket.id, draftFromTicket(ticket)])));
    } catch (err) {
      setMessage(err.message || "Could not load rework queue");
    }
  }

  function updateDraft(ticketId, field, value) {
    setDrafts((current) => ({
      ...current,
      [ticketId]: {
        ...(current[ticketId] || {}),
        [field]: value,
      },
    }));
  }

  async function saveTicket(ticket, overrides = {}) {
    const payload = { ...(drafts[ticket.id] || draftFromTicket(ticket)), ...overrides };
    if (["completed", "closed"].includes(payload.status) && !payload.resolution_notes.trim()) {
      setMessage("Add resolution notes before completing or closing a rework ticket.");
      return;
    }

    try {
      const updated = await updateReworkTicket(ticket.id, payload);
      setTickets((items) => items.map((item) => (item.id === updated.id ? updated : item)));
      setDrafts((current) => ({ ...current, [updated.id]: draftFromTicket(updated) }));
      setMessage(`${updated.ticket_number || "Ticket"} updated to ${statusLabel(updated.status)}`);
    } catch (err) {
      setMessage(err.message || "Could not update ticket");
    }
  }

  useEffect(() => {
    loadTickets();
  }, []);

  return (
    <AppShell title="Rework Queue" subtitle="Assign defective products for repair, track progress, and return completed items for final review.">
      <div className="page-actions">
        <button className="ghost-button" type="button" onClick={loadTickets}>
          <RefreshCw size={16} />
          Refresh
        </button>
        <label className="inline-control">
          Status
          <select value={status} onChange={(event) => setStatus(event.target.value)}>
            <option value="">All tickets</option>
            {statusOptions.map((option) => (
              <option key={option} value={option}>{statusLabel(option)}</option>
            ))}
          </select>
        </label>
        <button className="primary-button" type="button" onClick={loadTickets}>Apply</button>
        {message ? <span className={message.includes("Could not") || message.includes("Add") ? "inline-error" : "inline-success"}>{message}</span> : null}
      </div>

      <section className="ticket-grid">
        {tickets.map((ticket) => {
          const draft = drafts[ticket.id] || draftFromTicket(ticket);
          return (
            <article className="tool-panel ticket-card" key={ticket.id}>
              <div className="panel-heading">
                <div>
                  <h2>{ticket.ticket_number || "Rework ticket"}</h2>
                  <p>{ticket.product_id || "Unassigned Product"} | {ticket.batch_number || "No batch"} | {ticket.production_line || "No line"}</p>
                </div>
                <Wrench size={22} />
              </div>

              <div className="result-grid">
                <div className="metric-box">
                  <small>Status</small>
                  <strong>{statusLabel(ticket.status)}</strong>
                </div>
                <div className="metric-box">
                  <small>Priority</small>
                  <strong>{ticket.priority}</strong>
                </div>
                <div className="metric-box">
                  <small>Defect</small>
                  <strong>{ticket.defect_type || "Unknown"}</strong>
                </div>
                <div className="metric-box">
                  <small>Severity</small>
                  <SeverityBadge level={ticket.severity_level} />
                </div>
              </div>

              <div className="ticket-timeline">
                <span>Created {formatDateTime(ticket.created_at)}</span>
                {ticket.started_at ? <span>Started {formatDateTime(ticket.started_at)}</span> : null}
                {ticket.resolved_at ? <span>Resolved {formatDateTime(ticket.resolved_at)}</span> : null}
              </div>

              <div className="metadata-grid ticket-controls">
                <label>
                  Assigned to
                  <input
                    value={draft.assigned_to}
                    onChange={(event) => updateDraft(ticket.id, "assigned_to", event.target.value)}
                    placeholder="Supervisor or repair team"
                  />
                </label>
                <label>
                  Priority
                  <select value={draft.priority} onChange={(event) => updateDraft(ticket.id, "priority", event.target.value)}>
                    {priorityOptions.map((option) => (
                      <option key={option} value={option}>{option}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Status
                  <select value={draft.status} onChange={(event) => updateDraft(ticket.id, "status", event.target.value)}>
                    {statusOptions.map((option) => (
                      <option key={option} value={option}>{statusLabel(option)}</option>
                    ))}
                  </select>
                </label>
              </div>

              <label className="notes-field">
                Rework reason
                <textarea
                  value={draft.reason}
                  onChange={(event) => updateDraft(ticket.id, "reason", event.target.value)}
                  placeholder="Why this item was sent for rework"
                />
              </label>

              <label className="notes-field">
                Resolution notes
                <textarea
                  value={draft.resolution_notes}
                  onChange={(event) => updateDraft(ticket.id, "resolution_notes", event.target.value)}
                  placeholder="Repair action, retest notes, or closure reason"
                />
              </label>

              <div className="page-actions compact ticket-action-row">
                <button className="ghost-button" type="button" onClick={() => saveTicket(ticket, { status: "in_progress" })}>
                  <Wrench size={16} />
                  Start
                </button>
                <button className="ghost-button" type="button" onClick={() => saveTicket(ticket, { status: "completed" })}>
                  <CheckCircle2 size={16} />
                  Complete
                </button>
                <button className="primary-button" type="button" onClick={() => saveTicket(ticket)}>
                  <Save size={16} />
                  Save
                </button>
              </div>

              <small>Inspection {ticket.inspection_id}</small>
            </article>
          );
        })}
        {!tickets.length ? <div className="empty-panel">No rework tickets found.</div> : null}
      </section>
    </AppShell>
  );
}
