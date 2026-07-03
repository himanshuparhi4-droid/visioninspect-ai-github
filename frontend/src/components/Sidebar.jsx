"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  Camera,
  ClipboardList,
  Factory,
  FileText,
  Gauge,
  ImageUp,
  LogOut,
  ScanSearch,
  ShieldCheck,
  Wrench,
  Users,
} from "lucide-react";

import { logout } from "../services/authApi";

const navItems = [
  { href: "/dashboard", label: "Dashboard", icon: Gauge, roles: ["admin", "quality_manager", "factory_supervisor", "quality_engineer"] },
  { href: "/upload", label: "Inspect", icon: ImageUp, roles: ["admin", "quality_manager", "factory_supervisor", "quality_engineer"] },
  { href: "/camera", label: "Camera", icon: Camera, roles: ["admin", "quality_manager", "factory_supervisor", "quality_engineer"] },
  { href: "/inspection", label: "History", icon: ClipboardList, roles: ["admin", "quality_manager", "factory_supervisor", "quality_engineer"] },
  { href: "/rework", label: "Rework", icon: Wrench, roles: ["admin", "quality_manager", "factory_supervisor"] },
  { href: "/analytics", label: "Analytics", icon: BarChart3, roles: ["admin", "quality_manager", "factory_supervisor"] },
  { href: "/reports", label: "Reports", icon: FileText, roles: ["admin", "quality_manager", "factory_supervisor", "quality_engineer"] },
  { href: "/model-metrics", label: "Model", icon: ScanSearch, roles: ["admin", "quality_manager"] },
  { href: "/users", label: "Users", icon: Users, roles: ["admin", "quality_manager"] },
];

export default function Sidebar({ user }) {
  const pathname = usePathname();
  const role = user?.role || "quality_engineer";

  function handleLogout() {
    logout();
    window.location.href = "/login";
  }

  return (
    <aside className="sidebar">
      <Link href="/dashboard" className="brand" aria-label="VisionInspect AI dashboard">
        <span className="brand-mark">
          <Factory size={22} />
        </span>
        <span>
          <strong>VisionInspect AI</strong>
          <small>Quality inspection</small>
        </span>
      </Link>

      <nav className="side-nav" aria-label="Main navigation">
        {navItems.filter((item) => item.roles.includes(role)).map((item) => {
          const Icon = item.icon;
          const active = pathname === item.href;
          return (
            <Link key={item.href} className={active ? "nav-link active" : "nav-link"} href={item.href}>
              <Icon size={18} />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="security-chip">
          <ShieldCheck size={16} />
          <span>RBAC enabled</span>
        </div>
        <button className="ghost-button full-width" type="button" onClick={handleLogout}>
          <LogOut size={16} />
          <span>Logout</span>
        </button>
      </div>
    </aside>
  );
}
