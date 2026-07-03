"use client";

import { useEffect, useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";

import AnalyticsCards from "../../components/AnalyticsCards";
import AppShell from "../../components/AppShell";
import { downloadAnalyticsCsv, getAnalyticsSummary } from "../../services/analyticsApi";
import { getProductionCatalog } from "../../services/productionApi";

function Distribution({ title, values }) {
  const rows = Object.entries(values || {});
  const max = Math.max(...rows.map(([, value]) => value), 1);

  return (
    <section className="tool-panel">
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          <p>{rows.length || 0} categories</p>
        </div>
      </div>
      <div className="distribution-list">
        {rows.map(([label, value]) => (
          <div key={label} className="distribution-row">
            <span>{label}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${(value / max) * 100}%` }} />
            </div>
            <strong>{value}</strong>
          </div>
        ))}
        {!rows.length ? <div className="empty-visual">No analytics data yet.</div> : null}
      </div>
    </section>
  );
}

export default function AnalyticsPage() {
  const [summary, setSummary] = useState(null);
  const [filters, setFilters] = useState({ dateFrom: "", dateTo: "", productionLine: "", productId: "" });
  const [catalog, setCatalog] = useState({ products: [], production_lines: [] });
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  async function loadSummary() {
    setLoading(true);
    setSummary(await getAnalyticsSummary(filters));
    setLoading(false);
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
    loadSummary().catch(() => setLoading(false));
    getProductionCatalog().then(setCatalog).catch(() => setCatalog({ products: [], production_lines: [] }));
  }, []);

  const qualityMix = useMemo(() => ({
    pass: summary?.pass_count || 0,
    review: summary?.review_count || 0,
    fail: summary?.fail_count || 0,
  }), [summary]);

  return (
    <AppShell title="Analytics" subtitle="Production quality monitoring and defect trend summary.">
      <div className="page-actions">
        <button className="ghost-button" type="button" onClick={loadSummary}>
          <RefreshCw size={16} />
          {loading ? "Refreshing" : "Refresh"}
        </button>
        <button className="ghost-button" type="button" onClick={handleCsvExport}>
          Export CSV
        </button>
        {message ? <span className="inline-error">{message}</span> : null}
      </div>

      <section className="tool-panel">
        <div className="metadata-grid">
          <label>
            Date from
            <input type="datetime-local" value={filters.dateFrom} onChange={(event) => setFilters((current) => ({ ...current, dateFrom: event.target.value }))} />
          </label>
          <label>
            Date to
            <input type="datetime-local" value={filters.dateTo} onChange={(event) => setFilters((current) => ({ ...current, dateTo: event.target.value }))} />
          </label>
          <label>
            Production line
            <select value={filters.productionLine} onChange={(event) => setFilters((current) => ({ ...current, productionLine: event.target.value }))}>
              <option value="">All lines</option>
              {catalog.production_lines.map((line) => (
                <option key={line.line_id} value={line.line_id}>{line.name}</option>
              ))}
            </select>
          </label>
          <label>
            Product ID
            <select value={filters.productId} onChange={(event) => setFilters((current) => ({ ...current, productId: event.target.value }))}>
              <option value="">All products</option>
              {catalog.products.map((product) => (
                <option key={product.product_id} value={product.product_id}>{product.product_id}</option>
              ))}
            </select>
          </label>
          <button className="primary-button" type="button" onClick={loadSummary}>
            Apply filters
          </button>
        </div>
      </section>

      <AnalyticsCards summary={summary} />

      <div className="inspection-layout">
        <Distribution title="Defect Type Distribution" values={summary?.defect_type_distribution} />
        <Distribution title="Severity Distribution" values={summary?.severity_distribution} />
      </div>
      <div className="inspection-layout">
        <Distribution title="Quality Decision Mix" values={qualityMix} />
        <Distribution title="Production Line Distribution" values={summary?.production_line_distribution} />
      </div>
      <div className="inspection-layout">
        <Distribution title="Review Workflow Status" values={summary?.review_status_distribution} />
        <Distribution title="Image Source Mix" values={summary?.source_type_distribution} />
      </div>

      <section className="tool-panel">
        <div className="panel-heading">
          <div>
            <h2>Daily Quality Trend</h2>
            <p>{summary?.trend_by_day?.length || 0} days</p>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th>Total</th>
                <th>Defective</th>
                <th>Good</th>
                <th>Pass</th>
                <th>Review</th>
                <th>Fail</th>
                <th>Defect rate</th>
                <th>Pass rate</th>
              </tr>
            </thead>
            <tbody>
              {(summary?.trend_by_day || []).map((row) => (
                <tr key={row.date}>
                  <td>{row.date}</td>
                  <td>{row.total}</td>
                  <td>{row.defective}</td>
                  <td>{row.good}</td>
                  <td>{row.pass}</td>
                  <td>{row.review}</td>
                  <td>{row.fail}</td>
                  <td>{`${((row.defect_rate || 0) * 100).toFixed(1)}%`}</td>
                  <td>{`${((row.pass_rate || 0) * 100).toFixed(1)}%`}</td>
                </tr>
              ))}
              {!summary?.trend_by_day?.length ? (
                <tr>
                  <td colSpan="9">No trend data yet.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>

      <section className="tool-panel">
        <div className="panel-heading">
          <div>
            <h2>Defect Type By Line</h2>
            <p>Line-wise defect concentration.</p>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Production Line</th>
                <th>Good</th>
                <th>Broken Large</th>
                <th>Broken Small</th>
                <th>Contamination</th>
                <th>Unknown</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(summary?.defect_type_by_line || {}).map(([line, values]) => (
                <tr key={line}>
                  <td>{line}</td>
                  <td>{values.good || 0}</td>
                  <td>{values.broken_large || 0}</td>
                  <td>{values.broken_small || 0}</td>
                  <td>{values.contamination || 0}</td>
                  <td>{values.unknown_defect || values.unknown || 0}</td>
                </tr>
              ))}
              {!Object.keys(summary?.defect_type_by_line || {}).length ? (
                <tr>
                  <td colSpan="6">No line-wise defect data yet.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>
    </AppShell>
  );
}
