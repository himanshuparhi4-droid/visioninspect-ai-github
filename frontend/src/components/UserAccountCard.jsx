"use client";

import { UserRound } from "lucide-react";

import { formatDateTime } from "../services/dateTime";

export const ROLE_OPTIONS = [
  { value: "quality_engineer", label: "Quality engineer" },
  { value: "factory_supervisor", label: "Factory supervisor" },
  { value: "quality_manager", label: "Quality manager" },
  { value: "admin", label: "Admin" },
];

export const EMPTY_USER_FORM = {
  name: "",
  email: "",
  password: "",
  role: "quality_engineer",
};

export default function UserAccountCard({
  user,
  password = "",
  onPasswordChange,
  onUpdate,
  onApproval,
  onPasswordReset,
}) {
  return (
    <div className="user-card">
      <span className="stat-icon">
        <UserRound size={18} />
      </span>
      <strong>{user.name}</strong>
      <small>{user.email}</small>
      <span className="role-chip">{user.role}</span>
      <small>
        {user.is_active ? "Active" : "Inactive"} | Approval: {user.approval_status || "approved"}
      </small>
      <small>
        Requested role: {user.requested_role || user.role} | Last login:{" "}
        {user.last_login_at ? formatDateTime(user.last_login_at) : "Never"}
      </small>

      {user.approval_status === "pending" ? (
        <div className="page-actions compact">
          <button className="primary-button" type="button" onClick={() => onApproval(user, "approve")}>
            Approve
          </button>
          <button className="ghost-button" type="button" onClick={() => onApproval(user, "reject")}>
            Reject
          </button>
        </div>
      ) : null}

      <select value={user.role} onChange={(event) => onUpdate(user.id, { role: event.target.value })}>
        {ROLE_OPTIONS.map((role) => (
          <option key={role.value} value={role.value}>
            {role.label}
          </option>
        ))}
      </select>

      <div className="page-actions compact">
        <button
          className="ghost-button"
          type="button"
          onClick={() => onUpdate(user.id, { is_active: !user.is_active })}
        >
          {user.is_active ? "Deactivate" : "Reactivate"}
        </button>
      </div>

      <div className="password-reset-row">
        <input
          type="password"
          minLength={6}
          placeholder="New password"
          value={password}
          onChange={(event) => onPasswordChange(user.id, event.target.value)}
        />
        <button className="ghost-button" type="button" onClick={() => onPasswordReset(user)}>
          Reset
        </button>
      </div>
    </div>
  );
}
