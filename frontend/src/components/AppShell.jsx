"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";

import { getAuthToken, setAuthToken } from "../services/api";
import { getCurrentUser } from "../services/authApi";
import Navbar from "./Navbar";
import Sidebar, { navItems } from "./Sidebar";

export default function AppShell({ title, subtitle, children }) {
  const pathname = usePathname();
  const [user, setUser] = useState(null);
  const [ready, setReady] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setReady(false);

    if (!getAuthToken()) {
      setUser(null);
      window.location.replace("/login");
      return () => {
        cancelled = true;
      };
    }

    getCurrentUser()
      .then((authenticatedUser) => {
        if (cancelled) return;
        const page = navItems.find((item) => item.href === pathname);
        if (page && !page.roles.includes(authenticatedUser.role)) {
          window.location.replace("/dashboard");
          return;
        }
        setUser(authenticatedUser);
        setReady(true);
      })
      .catch(() => {
        if (cancelled) return;
        setAuthToken(null);
        window.location.replace("/login?reason=session_expired");
      });

    return () => {
      cancelled = true;
    };
  }, [pathname]);

  useEffect(() => {
    setMobileNavOpen(false);
  }, [pathname]);

  function handleSidebarToggle() {
    setSidebarCollapsed((current) => !current);
  }

  function handleMobileNavToggle() {
    setMobileNavOpen((current) => !current);
  }

  if (!ready) {
    return (
      <main className="loading-screen">
        <div className="loader" />
      </main>
    );
  }

  return (
    <div className={sidebarCollapsed ? "app-layout sidebar-collapsed" : "app-layout"}>
      <Sidebar
        user={user}
        collapsed={sidebarCollapsed}
        mobileOpen={mobileNavOpen}
        onNavigate={() => setMobileNavOpen(false)}
        onToggleCollapse={handleSidebarToggle}
        onToggleMobile={handleMobileNavToggle}
      />
      <div className="app-main">
        <Navbar title={title} subtitle={subtitle} user={user} />
        <main className="page-content">{children}</main>
      </div>
    </div>
  );
}
