"use client";

import { useEffect, useState } from "react";
import { KeyRound, RefreshCw, ShieldCheck, UserPlus } from "lucide-react";

import AppShell from "../../components/AppShell";
import UserAccountCard, { EMPTY_USER_FORM, ROLE_OPTIONS } from "../../components/UserAccountCard";
import { formatDateTime } from "../../services/dateTime";
import {
  approveUser,
  createUser,
  listAuditLogs,
  listUsers,
  rejectUser,
  resetUserPassword,
  updateUser,
} from "../../services/userApi";

export default function UsersPage() {
  const [users, setUsers] = useState([]);
  const [auditLogs, setAuditLogs] = useState([]);
  const [form, setForm] = useState(EMPTY_USER_FORM);
  const [passwords, setPasswords] = useState({});
  const [notice, setNotice] = useState({ type: "", text: "" });
  const [loading, setLoading] = useState(false);

  async function loadUsers() {
    setNotice({ type: "", text: "" });
    try {
      const [userList, logs] = await Promise.all([listUsers(), listAuditLogs({ limit: 12 })]);
      setUsers(userList);
      setAuditLogs(logs);
    } catch (err) {
      setNotice({ type: "error", text: err.message || "User list is limited to admin and supervisor roles" });
    }
  }

  function updateField(field, value) {
    setForm((current) => ({ ...current, [field]: value }));
  }

  async function handleCreateUser(event) {
    event.preventDefault();
    setNotice({ type: "", text: "" });
    setLoading(true);

    try {
      const user = await createUser(form);
      setUsers((current) => [user, ...current.filter((item) => item.id !== user.id)]);
      setForm(EMPTY_USER_FORM);
      setNotice({ type: "success", text: `Created ${user.email}` });
    } catch (err) {
      setNotice({ type: "error", text: err.message || "Could not create user" });
    } finally {
      setLoading(false);
    }
  }

  async function handleUserUpdate(userId, payload) {
    setNotice({ type: "", text: "" });
    try {
      const updated = await updateUser(userId, payload);
      setUsers((current) => current.map((user) => (user.id === updated.id ? updated : user)));
      setNotice({ type: "success", text: `Updated ${updated.email}` });
    } catch (err) {
      setNotice({ type: "error", text: err.message || "Could not update user" });
    }
  }

  async function handleApproval(user, action) {
    setNotice({ type: "", text: "" });
    try {
      const updated = action === "approve" ? await approveUser(user.id) : await rejectUser(user.id);
      setUsers((current) => current.map((item) => (item.id === updated.id ? updated : item)));
      setNotice({ type: "success", text: `${action === "approve" ? "Approved" : "Rejected"} ${updated.email}` });
    } catch (err) {
      setNotice({ type: "error", text: err.message || "Could not update approval" });
    }
  }

  async function handlePasswordReset(user) {
    const password = passwords[user.id] || "";
    if (!password) {
      setNotice({ type: "error", text: "Enter a new password first" });
      return;
    }
    try {
      const updated = await resetUserPassword(user.id, password);
      setPasswords((current) => ({ ...current, [user.id]: "" }));
      setNotice({ type: "success", text: `Password reset for ${updated.email}` });
    } catch (err) {
      setNotice({ type: "error", text: err.message || "Could not reset password" });
    }
  }

  function updatePassword(userId, password) {
    setPasswords((current) => ({ ...current, [userId]: password }));
  }

  useEffect(() => {
    loadUsers();
  }, []);

  return (
    <AppShell title="Users" subtitle="Quality engineer, supervisor, and admin account visibility.">
      <div className="page-actions">
        <button className="ghost-button" type="button" onClick={loadUsers}>
          <RefreshCw size={16} />
          Refresh
        </button>
        {notice.text ? (
          <span className={notice.type === "error" ? "inline-error" : "inline-success"}>{notice.text}</span>
        ) : null}
      </div>

      <div className="user-management-layout">
        <section className="tool-panel">
          <div className="panel-heading">
            <div>
              <h2>Create User</h2>
              <p>Role-based account setup</p>
            </div>
            <UserPlus size={22} />
          </div>
          <form className="compact-form" onSubmit={handleCreateUser}>
            <label>
              Name
              <input value={form.name} onChange={(event) => updateField("name", event.target.value)} required />
            </label>
            <label>
              Email
              <input
                type="email"
                value={form.email}
                onChange={(event) => updateField("email", event.target.value)}
                required
              />
            </label>
            <label>
              Role
              <select value={form.role} onChange={(event) => updateField("role", event.target.value)}>
                {ROLE_OPTIONS.map((role) => (
                  <option key={role.value} value={role.value}>
                    {role.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Password
              <input
                type="password"
                value={form.password}
                onChange={(event) => updateField("password", event.target.value)}
                minLength={6}
                required
              />
            </label>
            <button className="primary-button" type="submit" disabled={loading}>
              {loading ? <RefreshCw className="spin" size={16} /> : <KeyRound size={16} />}
              <span>{loading ? "Creating" : "Create User"}</span>
            </button>
          </form>
        </section>

        <section className="tool-panel">
          <div className="panel-heading">
            <div>
              <h2>User Management</h2>
              <p>{users.length} visible accounts</p>
            </div>
            <ShieldCheck size={22} />
          </div>
          <div className="user-grid">
            {users.map((user) => (
              <UserAccountCard
                key={user.id}
                user={user}
                password={passwords[user.id] || ""}
                onPasswordChange={updatePassword}
                onUpdate={handleUserUpdate}
                onApproval={handleApproval}
                onPasswordReset={handlePasswordReset}
              />
            ))}
            {!users.length ? <div className="empty-visual">No users visible for this role.</div> : null}
          </div>
        </section>
      </div>

      <section className="tool-panel">
        <div className="panel-heading">
          <div>
            <h2>Audit Log</h2>
            <p>{auditLogs.length} recent platform events</p>
          </div>
          <ShieldCheck size={22} />
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Action</th>
                <th>Entity</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody>
              {auditLogs.map((log) => (
                <tr key={log.id}>
                  <td>{formatDateTime(log.created_at)}</td>
                  <td>{log.action}</td>
                  <td>{log.entity_type}</td>
                  <td>
                    {Object.entries(log.metadata || {})
                      .map(([key, value]) => `${key}: ${value}`)
                      .join(", ") || "None"}
                  </td>
                </tr>
              ))}
              {!auditLogs.length ? (
                <tr>
                  <td colSpan="4">No audit logs visible.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>
    </AppShell>
  );
}
