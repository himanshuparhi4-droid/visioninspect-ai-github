"use client";

import { useEffect, useRef, useState } from "react";
import { Bell, ChevronDown } from "lucide-react";

import { logout } from "../services/authApi";

export default function Navbar({ title, subtitle, user }) {
  const [openMenu, setOpenMenu] = useState("");
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
            <span>8</span>
          </button>
          {openMenu === "notifications" ? (
            <div className="topbar-dropdown notification-dropdown">
              <strong>Notifications</strong>
              <p>8 quality events need attention.</p>
              <div className="notification-list">
                <span>
                  <b>Critical defects</b>
                  <small>24 critical inspections recorded</small>
                </span>
                <span>
                  <b>Rework queue</b>
                  <small>2 products are waiting for repair action</small>
                </span>
                <span>
                  <b>Model confidence</b>
                  <small>Average confidence is stable at 84.4%</small>
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
                <a href="/users">User management</a>
                <a href="/model-metrics">Model settings</a>
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
