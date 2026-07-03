"use client";

import { Activity, Database, Server } from "lucide-react";

export default function Navbar({ title, subtitle, user }) {
  return (
    <header className="topbar">
      <div>
        <h1>{title}</h1>
        {subtitle ? <p>{subtitle}</p> : null}
      </div>
      <div className="topbar-actions">
        <span className="status-pill">
          <Server size={15} />
          API live
        </span>
        <span className="status-pill">
          <Database size={15} />
          MongoDB
        </span>
        <span className="status-pill">
          <Activity size={15} />
          Model v1
        </span>
        {user ? (
          <span className="user-pill">
            <strong>{user.name}</strong>
            <small>{user.role}</small>
          </span>
        ) : null}
      </div>
    </header>
  );
}
