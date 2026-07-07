"use client";

import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, RefreshCw, TimerReset } from "lucide-react";

import AppShell from "../../components/AppShell";
import ReworkTicketCard, {
  REWORK_STATUS_OPTIONS,
  draftFromTicket,
  fromLocalInputValue,
  statusLabel,
} from "../../components/ReworkTicketCard";
import { listReworkTickets, updateReworkTicket } from "../../services/reworkApi";
import { listUsers } from "../../services/userApi";

export default function ReworkPage() {
  const [tickets, setTickets] = useState([]);
  const [drafts, setDrafts] = useState({});
  const [users, setUsers] = useState([]);
  const [status, setStatus] = useState("");
  const [message, setMessage] = useState("");
  const queueStats = useMemo(
    () => ({
      open: tickets.filter((ticket) => ticket.status === "open").length,
      progress: tickets.filter((ticket) => ticket.status === "in_progress").length,
      completed: tickets.filter((ticket) => ["completed", "closed"].includes(ticket.status)).length,
    }),
    [tickets]
  );

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
    const draft = drafts[ticket.id] || draftFromTicket(ticket);
    const payload = {
      ...draft,
      due_at: fromLocalInputValue(draft.due_at),
      ...overrides,
    };
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
    listUsers()
      .then((items) => setUsers(items.filter((user) => user.is_active && user.approval_status === "approved")))
      .catch(() => setUsers([]));
  }, []);

  return (
    <AppShell
      title="Rework Queue"
      subtitle="Assign defective products for repair, track progress, and return completed items for final review."
    >
      <div className="page-actions">
        <button className="ghost-button" type="button" onClick={loadTickets}>
          <RefreshCw size={16} />
          Refresh
        </button>
        <label className="inline-control">
          Status
          <select value={status} onChange={(event) => setStatus(event.target.value)}>
            <option value="">All tickets</option>
            {REWORK_STATUS_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {statusLabel(option)}
              </option>
            ))}
          </select>
        </label>
        <button className="primary-button" type="button" onClick={loadTickets}>
          Apply
        </button>
        {message ? (
          <span
            className={message.includes("Could not") || message.includes("Add") ? "inline-error" : "inline-success"}
          >
            {message}
          </span>
        ) : null}
      </div>

      <section className="queue-strip">
        <div className="pulse-card">
          <AlertTriangle size={20} />
          <span>
            <strong>{queueStats.open}</strong>
            <small>Open tickets</small>
          </span>
        </div>
        <div className="pulse-card">
          <TimerReset size={20} />
          <span>
            <strong>{queueStats.progress}</strong>
            <small>In progress</small>
          </span>
        </div>
        <div className="pulse-card">
          <CheckCircle2 size={20} />
          <span>
            <strong>{queueStats.completed}</strong>
            <small>Completed/closed</small>
          </span>
        </div>
      </section>

      <section className="ticket-grid">
        {tickets.map((ticket) => (
          <ReworkTicketCard
            key={ticket.id}
            ticket={ticket}
            draft={drafts[ticket.id] || draftFromTicket(ticket)}
            users={users}
            onDraftChange={updateDraft}
            onSave={saveTicket}
          />
        ))}
        {!tickets.length ? <div className="empty-panel">No rework tickets found.</div> : null}
      </section>
    </AppShell>
  );
}
