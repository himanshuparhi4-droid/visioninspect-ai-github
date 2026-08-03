"use client";

import { useEffect, useRef, useState } from "react";
import { Bell, ChevronDown } from "lucide-react";

import { logout } from "../services/authApi";
import { getAnalyticsSummary } from "../services/analyticsApi";

export default function Navbar({ title, subtitle, user }) {
  const [openMenu, setOpenMenu] = useState("");
  const [summary, setSummary] = useState(null);
  const topbarMenuRef = useRef(null);

  function toggleMenu(menu) {
    setOpenMenu((current) => (current === menu ? "" : menu));
  }

  function handleLogout() {
    logout();
    window.location.href = "/login";
  }

  useEffect(() => {
    function handlePointerDown(event) {
      if (!topbarMenuRef.current?.contains(event.target)) {
        setOpenMenu("");
      }
    }

    function handleKeyDown(event) {
      if (event.key === "Escape") {
        setOpenMenu("");
      }
    }

    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  useEffect(() => {
    if (!user) return;
    getAnalyticsSummary()
      .then(setSummary)
      .catch(() => setSummary(null));
  }, [user]);

  const criticalCount = summary?.critical_count || 0;
  const reworkCount = summary?.rework_queue || 0;
  const reviewCount = summary?.review_count || 0;
  const notificationCount = criticalCount + reworkCount + reviewCount;
  const canManageUsers = ["admin", "quality_manager"].includes(user?.role);

  return (
    <header className="topbar">
      <div>
        <h1>{title || "VisionInspect AI"}</h1>
        {subtitle ? <p>{subtitle}</p> : null}
      </div>
      <div className="topbar-actions" ref={topbarMenuRef}>
        <div className="topbar-menu-wrap">
          <button
            className="icon-link notification-button"
            type="button"
            aria-label="Notifications"
            onClick={() => toggleMenu("notifications")}
          >
            <Bell size={16} />
            {notificationCount ? <span>{notificationCount}</span> : null}
          </button>
          {openMenu === "notifications" ? (
            <div className="topbar-dropdown notification-dropdown">
              <strong>Notifications</strong>
              <p>{notificationCount} quality events need attention.</p>
              <div className="notification-list">
                <span>
                  <b>Critical defects</b>
                  <small>{criticalCount} critical inspections recorded</small>
                </span>
                <span>
                  <b>Rework queue</b>
                  <small>{reworkCount} products are waiting for repair action</small>
                </span>
                <span>
                  <b>Manual review</b>
                  <small>{reviewCount} inspections require a decision</small>
                </span>
              </div>
            </div>
          ) : null}
        </div>
        {user ? (
          <div className="topbar-menu-wrap">
            <button className="user-pill user-menu-button" type="button" onClick={() => toggleMenu("user")}>
              <strong>{user.name}</strong>
              <small>{user.role}</small>
              <ChevronDown size={14} />
            </button>
            {openMenu === "user" ? (
              <div className="topbar-dropdown user-dropdown">
                <strong>{user.name}</strong>
                <small>{user.email || user.role}</small>
                <div className="dropdown-divider" />
                {canManageUsers ? <a href="/users">User management</a> : null}
                {canManageUsers ? <a href="/model-metrics">Model settings</a> : null}
                <button type="button" onClick={handleLogout}>
                  Logout
                </button>
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    </header>
  );
}
