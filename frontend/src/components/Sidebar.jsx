"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  Camera,
  ChevronLeft,
  ChevronRight,
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
  {
    href: "/dashboard",
    label: "Dashboard",
    icon: Gauge,
    section: "",
    roles: ["admin", "quality_manager", "factory_supervisor", "quality_engineer"],
  },
  {
    href: "/upload",
    label: "Upload",
    icon: ImageUp,
    section: "Inspection",
    roles: ["admin", "quality_manager", "factory_supervisor", "quality_engineer"],
  },
  {
    href: "/camera",
    label: "Camera",
    icon: Camera,
    section: "Inspection",
    roles: ["admin", "quality_manager", "factory_supervisor", "quality_engineer"],
  },
  {
    href: "/inspection",
    label: "Inspection History",
    icon: ClipboardList,
    section: "Inspection",
    roles: ["admin", "quality_manager", "factory_supervisor", "quality_engineer"],
  },
  {
    href: "/rework",
    label: "Rework Queue",
    icon: Wrench,
    section: "Rework",
    roles: ["admin", "quality_manager", "factory_supervisor"],
  },
  {
    href: "/analytics",
    label: "Analytics",
    icon: BarChart3,
    section: "Reports & Analytics",
    roles: ["admin", "quality_manager", "factory_supervisor"],
  },
  {
    href: "/reports",
    label: "Reports",
    icon: FileText,
    section: "Reports & Analytics",
    roles: ["admin", "quality_manager", "factory_supervisor", "quality_engineer"],
  },
  {
    href: "/model-metrics",
    label: "Model Metrics",
    icon: ScanSearch,
    section: "Models",
    roles: ["admin", "quality_manager"],
  },
  { href: "/users", label: "Users", icon: Users, section: "Admin", roles: ["admin", "quality_manager"] },
];

export default function Sidebar({ user, collapsed = false, onToggleCollapse }) {
  const pathname = usePathname();
  const role = user?.role || "quality_engineer";

  function handleLogout() {
    logout();
    window.location.href = "/login";
  }

  const visibleItems = navItems.filter((item) => item.roles.includes(role));
  const sections = visibleItems.reduce((groups, item) => {
    const section = item.section || "Main";
    return { ...groups, [section]: [...(groups[section] || []), item] };
  }, {});

  const ToggleIcon = collapsed ? ChevronRight : ChevronLeft;

  return (
    <aside className={collapsed ? "sidebar collapsed" : "sidebar"}>
      <Link href="/dashboard" className="brand" aria-label="VisionInspect AI dashboard" data-label="VisionInspect AI">
        <span className="brand-mark">
          <Factory size={22} />
        </span>
        <span>
          <strong>VisionInspect AI</strong>
          <small>Quality inspection</small>
        </span>
      </Link>

      <nav className="side-nav" aria-label="Main navigation">
        {Object.entries(sections).map(([section, items]) => (
          <div key={section} className="nav-section">
            {section !== "Main" ? <span className="nav-section-label">{section}</span> : null}
            {items.map((item) => {
              const Icon = item.icon;
              const active = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  className={active ? "nav-link active" : "nav-link"}
                  href={item.href}
                  data-label={item.label}
                  title={collapsed ? item.label : undefined}
                >
                  <Icon size={18} />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="security-chip" data-label="RBAC enabled">
          <ShieldCheck size={16} />
          <span>RBAC enabled</span>
        </div>
        <button className="ghost-button full-width" type="button" onClick={handleLogout} data-label="Logout">
          <LogOut size={16} />
          <span>Logout</span>
        </button>
        <button
          className="ghost-button full-width collapse-button"
          type="button"
          onClick={onToggleCollapse}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          data-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <ToggleIcon size={16} />
          <span>{collapsed ? "Expand" : "Collapse"}</span>
        </button>
      </div>
    </aside>
  );
}
