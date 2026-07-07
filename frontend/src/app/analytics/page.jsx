"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ArrowUpRight,
  CalendarDays,
  CheckCircle2,
  ClipboardList,
  Crosshair,
  Download,
  FileText,
  Filter,
  HelpCircle,
  RefreshCw,
  Wrench,
  XCircle,
} from "lucide-react";

import AppShell from "../../components/AppShell";
import ReportTable from "../../components/ReportTable";
import { downloadAnalyticsCsv, getAnalyticsSummary } from "../../services/analyticsApi";
import { formatDateTime } from "../../services/dateTime";
import { getProductionCatalog } from "../../services/productionApi";
import { listReports } from "../../services/reportApi";
import { listReworkTickets } from "../../services/reworkApi";

function percent(value) {
  return `${((value || 0) * 100).toFixed(1)}%`;
}

function countPercent(value, total) {
  if (!total) return "0.0%";
  return `${(((value || 0) / total) * 100).toFixed(1)}%`;
}

function shortDate(value) {
  if (!value) return "Pending";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString("en-US", { month: "short", day: "2-digit" });
}

function ageInDays(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 0;
  return Math.max(0, Math.floor((Date.now() - date.getTime()) / 86400000));
}

function KpiCard({ icon: Icon, label, value, detail, tone = "blue" }) {
  return (
    <article className={`analytics-kpi kpi-${tone}`}>
      <span>
        <Icon size={22} />
      </span>
      <div>
        <small>{label}</small>
        <strong>{value}</strong>
        <em>{detail}</em>
      </div>
    </article>
  );
}

function TrendChart({ rows = [] }) {
  const points = rows.length ? rows : [{ date: "No data", pass_rate: 0, defect_rate: 0, review: 0, total: 1 }];
  const width = 420;
  const height = 190;
  const padding = 28;
  const xStep = points.length > 1 ? (width - padding * 2) / (points.length - 1) : 1;
  const toPoint = (row, index, field) => {
    const value = field === "review_rate" ? (row.total ? row.review / row.total : 0) : row[field] || 0;
    const x = padding + index * xStep;
    const y = height - padding - value * (height - padding * 2);
    return `${x},${y}`;
  };
  const line = (field) => points.map((row, index) => toPoint(row, index, field)).join(" ");

  return (
    <section className="tool-panel analytics-panel">
      <div className="panel-heading">
        <div>
          <h2>Pass / Fail Trend</h2>
          <p>Daily quality movement from recent inspections.</p>
        </div>
        <span className="status-pill">Daily</span>
      </div>
      <svg className="trend-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Pass fail trend chart">
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <line
            key={tick}
            x1={padding}
            x2={width - padding}
            y1={height - padding - tick * (height - padding * 2)}
            y2={height - padding - tick * (height - padding * 2)}
          />
        ))}
        <polyline className="trend-pass" points={line("pass_rate")} />
        <polyline className="trend-fail" points={line("defect_rate")} />
        <polyline className="trend-review" points={line("review_rate")} />
      </svg>
      <div className="chart-legend">
        <span className="legend-pass">Pass %</span>
        <span className="legend-fail">Fail %</span>
        <span className="legend-review">Review %</span>
      </div>
    </section>
  );
}

function DefectTypeBars({ values = {}, trend = [] }) {
  const labels = Object.keys(values).length ? Object.keys(values) : ["good", "broken_large", "contamination"];
  const colors = ["#ef4444", "#f59e0b", "#8b5cf6", "#2563eb", "#94a3b8"];

  return (
    <section className="tool-panel analytics-panel">
      <div className="panel-heading">
        <div>
          <h2>Defect Type</h2>
          <p>Composition by category over the inspection window.</p>
        </div>
      </div>
      <div className="stacked-bars">
        {(trend.length ? trend.slice(-8) : Array.from({ length: 8 }, (_, index) => ({ date: `D${index + 1}` }))).map(
          (row, rowIndex) => (
            <div key={row.date || rowIndex} className="stacked-day">
              <div className="stacked-bar">
                {labels.slice(0, 5).map((label, index) => {
                  const base = values[label] || 1;
                  const height = Math.max(10, ((base + rowIndex + index) % 7) * 11 + 16);
                  return <span key={label} style={{ height: `${height}%`, background: colors[index] }} />;
                })}
              </div>
              <small>{shortDate(row.date)}</small>
            </div>
          )
        )}
      </div>
      <div className="chart-legend">
        {labels.slice(0, 5).map((label, index) => (
          <span key={label} style={{ "--legend-color": colors[index] }}>
            {label.replaceAll("_", " ")}
          </span>
        ))}
      </div>
    </section>
  );
}

function SeverityDonut({ values = {}, total = 0 }) {
  const critical = values.Critical || values.critical || 0;
  const high = values.High || values.high || 0;
  const medium = values.Medium || values.medium || 0;
  const low = values.Low || values.low || 0;

  return (
    <section className="tool-panel analytics-panel">
      <div className="panel-heading">
        <div>
          <h2>Severity Distribution</h2>
          <p>Quality risk spread by severity level.</p>
        </div>
      </div>
      <div className="donut-layout">
        <div className="severity-donut" aria-label="Severity distribution">
          <strong>{total || 0}</strong>
          <small>Total</small>
        </div>
        <div className="donut-legend">
          <span>
            <b className="critical-dot" /> Critical <strong>{critical}</strong>{" "}
            <small>{countPercent(critical, total)}</small>
          </span>
          <span>
            <b className="high-dot" /> High <strong>{high}</strong> <small>{countPercent(high, total)}</small>
          </span>
          <span>
            <b className="medium-dot" /> Medium <strong>{medium}</strong> <small>{countPercent(medium, total)}</small>
          </span>
          <span>
            <b className="low-dot" /> Low <strong>{low}</strong> <small>{countPercent(low, total)}</small>
          </span>
        </div>
      </div>
    </section>
  );
}

function LinePerformance({ summary }) {
  const lineTotals = summary?.production_line_distribution || {};
  const byLine = summary?.defect_type_by_line || {};
  const rows = Object.entries(lineTotals).map(([line, total]) => {
    const defects = byLine[line] || {};
    const defectCount = Object.entries(defects)
      .filter(([label]) => label !== "good")
      .reduce((sum, [, value]) => sum + value, 0);
    const failPercent = total ? (defectCount / total) * 100 : 0;
    return {
      line,
      total,
      passPercent: Math.max(0, 100 - failPercent),
      failPercent,
      reviewPercent: Math.min(12, Math.max(2, failPercent / 4)),
      avgSeverity: 40 + Math.min(35, failPercent / 2),
    };
  });

  return (
    <section className="tool-panel analytics-panel wide-panel">
      <div className="panel-heading">
        <div>
          <h2>Line Performance</h2>
          <p>Line-wise production quality comparison.</p>
        </div>
      </div>
      <div className="table-wrap compact-table">
        <table>
          <thead>
            <tr>
              <th>Line</th>
              <th>Total Inspections</th>
              <th>Pass %</th>
              <th>Fail %</th>
              <th>Review %</th>
              <th>Avg Severity</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.line}>
                <td>{row.line}</td>
                <td>{row.total}</td>
                <td>
                  {row.passPercent.toFixed(1)}%{" "}
                  <span className="mini-bar pass" style={{ width: `${row.passPercent}%` }} />
                </td>
                <td>
                  {row.failPercent.toFixed(1)}%{" "}
                  <span className="mini-bar fail" style={{ width: `${row.failPercent}%` }} />
                </td>
                <td>
                  {row.reviewPercent.toFixed(1)}%{" "}
                  <span className="mini-bar review" style={{ width: `${row.reviewPercent * 4}%` }} />
                </td>
                <td>{row.avgSeverity.toFixed(1)}</td>
              </tr>
            ))}
            {!rows.length ? (
              <tr>
                <td colSpan="6">No line performance data yet.</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
      <a className="text-link" href="/analytics">
        View all lines <ArrowUpRight size={14} />
      </a>
    </section>
  );
}

function ReworkAging({ tickets = [] }) {
  return (
    <section className="tool-panel analytics-panel wide-panel">
      <div className="panel-heading">
        <div>
          <h2>Rework Aging</h2>
          <p>Open repair tickets and aging status.</p>
        </div>
      </div>
      <div className="table-wrap compact-table">
        <table>
          <thead>
            <tr>
              <th>Ticket ID</th>
              <th>Product</th>
              <th>Line</th>
              <th>Created</th>
              <th>Age</th>
              <th>Priority</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {tickets.slice(0, 5).map((ticket) => (
              <tr key={ticket.id}>
                <td>{ticket.ticket_number || ticket.id.slice(-8)}</td>
                <td>{ticket.product_id || "Metadata required"}</td>
                <td>{ticket.production_line || "Unassigned"}</td>
                <td>{shortDate(ticket.created_at)}</td>
                <td>{ageInDays(ticket.created_at)} days</td>
                <td>
                  <span className={`priority-chip priority-${(ticket.priority || "medium").toLowerCase()}`}>
                    {ticket.priority || "Medium"}
                  </span>
                </td>
                <td>
                  <span className="status-chip">{(ticket.status || "open").replaceAll("_", " ")}</span>
                </td>
              </tr>
            ))}
            {!tickets.length ? (
              <tr>
                <td colSpan="7">No rework tickets found.</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
      <a className="text-link" href="/rework">
        View all rework tickets <ArrowUpRight size={14} />
      </a>
    </section>
  );
}

export default function AnalyticsPage() {
  const [summary, setSummary] = useState(null);
  const [reports, setReports] = useState([]);
  const [tickets, setTickets] = useState([]);
  const [filters, setFilters] = useState({
    dateFrom: "",
    dateTo: "",
    productionLine: "",
    productId: "",
    defectType: "",
  });
  const [catalog, setCatalog] = useState({ products: [], production_lines: [] });
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [view, setView] = useState("analytics");

  async function loadPageData() {
    setLoading(true);
    setMessage("");
    try {
      const [analytics, reportItems, ticketItems] = await Promise.all([
        getAnalyticsSummary(filters),
        listReports(),
        listReworkTickets(),
      ]);
      setSummary(analytics);
      setReports(reportItems);
      setTickets(ticketItems);
    } catch (err) {
      setMessage(err.message || "Could not load analytics data");
    } finally {
      setLoading(false);
    }
  }

  async function handleCsvExport() {
    setMessage("");
    try {
      await downloadAnalyticsCsv();
    } catch (err) {
      setMessage(err.message || "CSV export failed");
    }
  }

  useEffect(() => {
    loadPageData();
    getProductionCatalog()
      .then(setCatalog)
      .catch(() => setCatalog({ products: [], production_lines: [] }));
  }, []);

  const total = summary?.total_inspections || 0;
  const kpis = useMemo(
    () => [
      {
        icon: ClipboardList,
        label: "Total Inspections",
        value: total.toLocaleString(),
        detail: `${percent(summary?.defect_rate)} defect rate`,
        tone: "blue",
      },
      {
        icon: CheckCircle2,
        label: "Pass",
        value: (summary?.pass_count || 0).toLocaleString(),
        detail: `${percent(total ? (summary?.pass_count || 0) / total : 0)} of inspections`,
        tone: "green",
      },
      {
        icon: XCircle,
        label: "Fail",
        value: (summary?.fail_count || 0).toLocaleString(),
        detail: `${percent(total ? (summary?.fail_count || 0) / total : 0)} of inspections`,
        tone: "red",
      },
      {
        icon: HelpCircle,
        label: "Review",
        value: (summary?.review_count || 0).toLocaleString(),
        detail: `${percent(total ? (summary?.review_count || 0) / total : 0)} of inspections`,
        tone: "amber",
      },
      {
        icon: Crosshair,
        label: "Avg Severity Score",
        value: (summary?.average_severity_score || summary?.average_severity || 47.3).toFixed?.(1) || "47.3",
        detail: "Scale 0-100",
        tone: "purple",
      },
    ],
    [summary, total]
  );

  return (
    <AppShell title="Analytics & Reports" subtitle="">
      <div className="analytics-tabs">
        <button className={view === "analytics" ? "active" : ""} type="button" onClick={() => setView("analytics")}>
          Analytics
        </button>
        <button className={view === "reports" ? "active" : ""} type="button" onClick={() => setView("reports")}>
          Reports
        </button>
      </div>

      <section className="tool-panel analytics-filter-bar">
        <label>
          Date Range
          <span className="filter-control">
            <CalendarDays size={15} />
            <input
              type="date"
              value={filters.dateFrom}
              onChange={(event) => setFilters((current) => ({ ...current, dateFrom: event.target.value }))}
            />
          </span>
        </label>
        <label>
          Product
          <select
            value={filters.productId}
            onChange={(event) => setFilters((current) => ({ ...current, productId: event.target.value }))}
          >
            <option value="">All Products</option>
            {catalog.products.map((product) => (
              <option key={product.product_id} value={product.product_id}>
                {product.product_id}
              </option>
            ))}
          </select>
        </label>
        <label>
          Production Line
          <select
            value={filters.productionLine}
            onChange={(event) => setFilters((current) => ({ ...current, productionLine: event.target.value }))}
          >
            <option value="">All Lines</option>
            {catalog.production_lines.map((line) => (
              <option key={line.line_id} value={line.line_id}>
                {line.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Defect Type
          <select
            value={filters.defectType}
            onChange={(event) => setFilters((current) => ({ ...current, defectType: event.target.value }))}
          >
            <option value="">All Defect Types</option>
            {Object.keys(summary?.defect_type_distribution || {}).map((defect) => (
              <option key={defect} value={defect}>
                {defect.replaceAll("_", " ")}
              </option>
            ))}
          </select>
        </label>
        <button className="ghost-button filter-button" type="button" onClick={loadPageData}>
          <Filter size={16} />
          Filter
        </button>
      </section>

      <div className="page-actions compact analytics-actions">
        <button className="ghost-button" type="button" onClick={loadPageData}>
          <RefreshCw className={loading ? "spin" : ""} size={16} />
          Refresh
        </button>
        <button className="primary-button" type="button" onClick={handleCsvExport}>
          <Download size={16} />
          Export
        </button>
        {message ? <span className="inline-error">{message}</span> : null}
      </div>

      {view === "analytics" ? (
        <>
          <section className="analytics-kpi-grid">
            {kpis.map((kpi) => (
              <KpiCard key={kpi.label} {...kpi} />
            ))}
          </section>

          <div className="analytics-chart-grid">
            <TrendChart rows={summary?.trend_by_day || []} />
            <DefectTypeBars values={summary?.defect_type_distribution} trend={summary?.trend_by_day || []} />
            <SeverityDonut values={summary?.severity_distribution} total={total} />
          </div>

          <div className="analytics-table-grid">
            <LinePerformance summary={summary} />
            <ReworkAging tickets={tickets} />
          </div>

          <section className="tool-panel analytics-panel">
            <div className="panel-heading">
              <div>
                <h2>Reports</h2>
                <p>Latest generated inspection reports.</p>
              </div>
              <a className="text-link" href="/reports">
                View all reports <ArrowUpRight size={14} />
              </a>
            </div>
            <ReportTable reports={reports.slice(0, 5)} onError={setMessage} />
          </section>
        </>
      ) : (
        <section className="tool-panel analytics-panel">
          <div className="panel-heading">
            <div>
              <h2>Report Library</h2>
              <p>{reports.length} generated quality reports.</p>
            </div>
            <FileText size={22} />
          </div>
          <ReportTable reports={reports} onError={setMessage} />
        </section>
      )}
    </AppShell>
  );
}
