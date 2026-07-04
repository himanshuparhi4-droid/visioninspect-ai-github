"use client";

import Link from "next/link";
import { CheckCircle2, RefreshCw, Save, Wrench } from "lucide-react";

import SeverityBadge from "./SeverityBadge";
import { formatDateTime } from "../services/dateTime";

export const REWORK_STATUS_OPTIONS = ["open", "in_progress", "completed", "closed"];
export const REWORK_PRIORITY_OPTIONS = ["Low", "Medium", "High", "Critical"];

export function statusLabel(value = "") {
  return value.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

export function toLocalInputValue(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const pad = (part) => String(part).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

export function fromLocalInputValue(value) {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date.toISOString();
}

export function draftFromTicket(ticket) {
  return {
    assigned_to: ticket.assigned_to || "",
    priority: ticket.priority || "Medium",
    status: ticket.status || "open",
    reason: ticket.reason || "",
    resolution_notes: ticket.resolution_notes || "",
    due_at: toLocalInputValue(ticket.due_at),
  };
}

function ageInDays(start, end = null) {
  const startDate = new Date(start);
  const endDate = end ? new Date(end) : new Date();
  if (Number.isNaN(startDate.getTime()) || Number.isNaN(endDate.getTime())) return 0;
  return Math.max(0, Math.floor((endDate - startDate) / 86400000));
}

function isOverdue(ticket) {
  if (!ticket.due_at || ["completed", "closed"].includes(ticket.status)) return false;
  return new Date(ticket.due_at).getTime() < Date.now();
}

export default function ReworkTicketCard({ ticket, draft, users = [], onDraftChange, onSave }) {
  const assignedInUserList = users.some((user) => user.email === draft.assigned_to);

  function update(field, value) {
    onDraftChange(ticket.id, field, value);
  }

  return (
    <article className="tool-panel ticket-card">
      <div className="panel-heading">
        <div>
          <h2>{ticket.ticket_number || "Rework ticket"}</h2>
          <p>
            {ticket.product_id || "Unassigned Product"} | {ticket.batch_number || "No batch"} |{" "}
            {ticket.production_line || "No line"}
          </p>
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
        <span>{ageInDays(ticket.created_at, ticket.resolved_at)} day aging</span>
        {ticket.due_at ? (
          <span className={isOverdue(ticket) ? "overdue-chip" : ""}>Due {formatDateTime(ticket.due_at)}</span>
        ) : null}
        {ticket.started_at ? <span>Started {formatDateTime(ticket.started_at)}</span> : null}
        {ticket.resolved_at ? <span>Resolved {formatDateTime(ticket.resolved_at)}</span> : null}
      </div>

      <div className="metadata-grid ticket-controls">
        <label>
          Assigned to
          <select value={draft.assigned_to} onChange={(event) => update("assigned_to", event.target.value)}>
            <option value="">Unassigned</option>
            {draft.assigned_to && !assignedInUserList ? (
              <option value={draft.assigned_to}>{draft.assigned_to}</option>
            ) : null}
            {users.map((user) => (
              <option key={user.id} value={user.email}>
                {user.name} ({user.role.replaceAll("_", " ")})
              </option>
            ))}
          </select>
        </label>
        <label>
          Priority
          <select value={draft.priority} onChange={(event) => update("priority", event.target.value)}>
            {REWORK_PRIORITY_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
        <label>
          Status
          <select value={draft.status} onChange={(event) => update("status", event.target.value)}>
            {REWORK_STATUS_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {statusLabel(option)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Due date
          <input
            type="datetime-local"
            value={draft.due_at}
            onChange={(event) => update("due_at", event.target.value)}
          />
        </label>
      </div>

      <label className="notes-field">
        Rework reason
        <textarea
          value={draft.reason}
          onChange={(event) => update("reason", event.target.value)}
          placeholder="Why this item was sent for rework"
        />
      </label>

      <label className="notes-field">
        Resolution notes
        <textarea
          value={draft.resolution_notes}
          onChange={(event) => update("resolution_notes", event.target.value)}
          placeholder="Repair action, retest notes, or closure reason"
        />
      </label>

      <div className="page-actions compact ticket-action-row">
        <button className="ghost-button" type="button" onClick={() => onSave(ticket, { status: "in_progress" })}>
          <Wrench size={16} />
          Start
        </button>
        <button className="ghost-button" type="button" onClick={() => onSave(ticket, { status: "completed" })}>
          <CheckCircle2 size={16} />
          Complete
        </button>
        <button className="primary-button" type="button" onClick={() => onSave(ticket)}>
          <Save size={16} />
          Save
        </button>
        {["completed", "closed"].includes(ticket.status) ? (
          <Link className="ghost-button" href={`/upload?rework_ticket=${ticket.id}&inspection=${ticket.inspection_id}`}>
            <RefreshCw size={16} />
            Re-inspect
          </Link>
        ) : null}
      </div>

      <small>Inspection {ticket.inspection_id}</small>
    </article>
  );
}
