"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowUpRight, RefreshCw } from "lucide-react";

import AnalyticsCards from "../../components/AnalyticsCards";
import AppShell from "../../components/AppShell";
import SeverityBadge from "../../components/SeverityBadge";
import { getAnalyticsSummary } from "../../services/analyticsApi";
import { formatDateTime } from "../../services/dateTime";
import { listInspections } from "../../services/inspectionApi";

export default function DashboardPage() {
  const [summary, setSummary] = useState(null);
  const [inspections, setInspections] = useState([]);
  const [loading, setLoading] = useState(true);

  async function loadDashboard() {
    setLoading(true);
    const [analytics, history] = await Promise.all([
      getAnalyticsSummary(),
      listInspections({ limit: 6 }),
    ]);
    setSummary(analytics);
    setInspections(history.items || []);
    setLoading(false);
  }

  useEffect(() => {
    loadDashboard().catch(() => setLoading(false));
  }, []);

  return (
    <AppShell title="Dashboard" subtitle="Production quality overview and latest inspection outcomes.">
      <div className="page-actions">
        <button className="ghost-button" type="button" onClick={loadDashboard}>
          <RefreshCw size={16} />
          Refresh
        </button>
        <Link className="primary-link" href="/upload">
          New inspection
          <ArrowUpRight size={16} />
        </Link>
      </div>

      <AnalyticsCards summary={summary} />

      <section className="tool-panel">
        <div className="panel-heading">
          <div>
            <h2>Recent Inspections</h2>
            <p>{loading ? "Loading records..." : `${inspections.length} latest records`}</p>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Prediction</th>
                <th>Product</th>
                <th>Line</th>
                <th>Defect</th>
                <th>Severity</th>
                <th>Decision</th>
                <th>Confidence</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {inspections.map((item) => (
                <tr key={item.id}>
                  <td>{item.prediction || "Pending"}</td>
                  <td>{item.product_id || "Unassigned"}</td>
                  <td>{item.production_line || "Unassigned"}</td>
                  <td>{item.defect_type || "Unknown"}</td>
                  <td><SeverityBadge level={item.severity_level} /></td>
                  <td>{item.pass_fail || "Pending"}</td>
                  <td>{item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : "Pending"}</td>
                  <td>{formatDateTime(item.created_at)}</td>
                </tr>
              ))}
              {!inspections.length ? (
                <tr>
                  <td colSpan="8">No inspections yet.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>
    </AppShell>
  );
}
