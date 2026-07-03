"use client";

import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";

import AppShell from "../../components/AppShell";
import ReportTable from "../../components/ReportTable";
import { listReports } from "../../services/reportApi";

export default function ReportsPage() {
  const [reports, setReports] = useState([]);
  const [message, setMessage] = useState("");

  async function loadReports() {
    setMessage("");
    setReports(await listReports());
  }

  useEffect(() => {
    loadReports().catch(() => setReports([]));
  }, []);

  return (
    <AppShell title="Reports" subtitle="Generated inspection reports and quality records.">
      <div className="page-actions">
        <button className="ghost-button" type="button" onClick={loadReports}>
          <RefreshCw size={16} />
          Refresh
        </button>
        {message ? <span className="inline-error">{message}</span> : null}
      </div>
      <section className="tool-panel">
        <div className="panel-heading">
          <div>
            <h2>Report Registry</h2>
            <p>{reports.length} reports</p>
          </div>
        </div>
        <ReportTable reports={reports} onError={setMessage} />
      </section>
    </AppShell>
  );
}
