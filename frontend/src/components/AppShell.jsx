"use client";

import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";

import { getAuthToken, setAuthToken } from "../services/api";
import { getCurrentUser } from "../services/authApi";
import Navbar from "./Navbar";
import Sidebar, { navItems } from "./Sidebar";

export default function AppShell({ title, subtitle, children }) {
  const router = useRouter();
  const pathname = usePathname();
  const [user, setUser] = useState(null);
  const [ready, setReady] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  useEffect(() => {
    if (!getAuthToken()) {
      router.replace("/login");
      return;
    }

    getCurrentUser()
      .then((authenticatedUser) => {
        const page = navItems.find((item) => item.href === pathname);
        if (page && !page.roles.includes(authenticatedUser.role)) {
          router.replace("/dashboard");
          return;
        }
        setUser(authenticatedUser);
        setReady(true);
      })
      .catch(() => {
        setAuthToken(null);
        router.replace("/login");
      });
  }, [pathname, router]);

  function handleSidebarToggle() {
    setSidebarCollapsed((current) => !current);
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
      <Sidebar user={user} collapsed={sidebarCollapsed} onToggleCollapse={handleSidebarToggle} />
      <div className="app-main">
        <Navbar title={title} subtitle={subtitle} user={user} />
        <main className="page-content">{children}</main>
      </div>
    </div>
  );
}
