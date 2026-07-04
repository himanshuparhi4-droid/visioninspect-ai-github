"use client";

import { useEffect, useRef, useState } from "react";
import { Camera, Play, RefreshCw, Square } from "lucide-react";

import AppShell from "../../components/AppShell";
import DefectHeatmap from "../../components/DefectHeatmap";
import InspectionResult from "../../components/InspectionResult";
import { formatTime } from "../../services/dateTime";
import { getCameraSamples, simulateCameraInspection } from "../../services/inspectionApi";

export default function CameraPage() {
  const [samples, setSamples] = useState(null);
  const [label, setLabel] = useState("");
  const [running, setRunning] = useState(false);
  const [frameIndex, setFrameIndex] = useState(0);
  const [feed, setFeed] = useState([]);
  const [selected, setSelected] = useState(null);
  const [message, setMessage] = useState("");
  const timerRef = useRef(null);
  const runningRef = useRef(false);
  const inFlightRef = useRef(false);
  const frameIndexRef = useRef(0);

  async function loadSamples() {
    setSamples(await getCameraSamples());
  }

  async function inspectNextFrame(nextIndex = frameIndexRef.current) {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    setMessage("");
    try {
      const inspection = await simulateCameraInspection({ frameIndex: nextIndex, label });
      setSelected(inspection);
      setFeed((current) => [inspection, ...current].slice(0, 12));
      frameIndexRef.current = nextIndex + 1;
      setFrameIndex(frameIndexRef.current);
    } catch (err) {
      setMessage(err.message || "Camera simulation failed");
      stopStream();
    } finally {
      inFlightRef.current = false;
    }
  }

  function chooseDemoLabel(nextLabel) {
    if (running) return;
    setLabel(nextLabel);
    frameIndexRef.current = 0;
    setFrameIndex(0);
    setFeed([]);
    setSelected(null);
  }

  function startStream() {
    if (runningRef.current) return;
    runningRef.current = true;
    setRunning(true);
    inspectNextFrame(frameIndexRef.current);
    timerRef.current = window.setInterval(() => {
      inspectNextFrame(frameIndexRef.current);
    }, 5000);
  }

  function stopStream() {
    runningRef.current = false;
    setRunning(false);
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }

  useEffect(() => {
    loadSamples().catch(() => setMessage("Could not load camera samples"));
    return () => {
      stopStream();
    };
  }, []);

  return (
    <AppShell
      title="Camera Simulation"
      subtitle="Simulated production-line image acquisition and live inspection feed."
    >
      <div className="page-actions">
        <button className="ghost-button" type="button" onClick={loadSamples}>
          <RefreshCw size={16} />
          Refresh samples
        </button>
        <button className="primary-button" type="button" onClick={startStream} disabled={running}>
          <Play size={16} />
          Start stream
        </button>
        <button
          className="ghost-button"
          type="button"
          onClick={() => inspectNextFrame(frameIndexRef.current)}
          disabled={running}
        >
          Inspect one frame
        </button>
        <button className="ghost-button" type="button" onClick={stopStream} disabled={!running}>
          <Square size={16} />
          Stop
        </button>
        {message ? <span className="inline-error">{message}</span> : null}
      </div>

      <section className="tool-panel">
        <div className="panel-heading">
          <div>
            <h2>Camera Source</h2>
            <p>{samples ? `${samples.total} sample frames available` : "Loading sample frames"}</p>
          </div>
          <Camera size={22} />
        </div>
        <div className="metadata-grid">
          <label>
            Defect stream
            <select value={label} onChange={(event) => chooseDemoLabel(event.target.value)} disabled={running}>
              <option value="">All sample frames</option>
              {Object.entries(samples?.labels || {}).map(([name, count]) => (
                <option key={name} value={name}>
                  {name} ({count})
                </option>
              ))}
            </select>
          </label>
          <div className="metric-box">
            <small>Stream status</small>
            <strong>{running ? "Running" : "Stopped"}</strong>
          </div>
          <div className="metric-box">
            <small>Frames inspected</small>
            <strong>{feed.length}</strong>
          </div>
          <div className="metric-box">
            <small>Next frame</small>
            <strong>{frameIndex}</strong>
          </div>
        </div>
        <div className="page-actions compact demo-controls">
          {(samples?.demo_controls || []).map((control) => (
            <button
              key={control.label}
              className={label === control.value ? "primary-button" : "ghost-button"}
              type="button"
              onClick={() => chooseDemoLabel(control.value)}
              disabled={running}
            >
              {control.label}
            </button>
          ))}
        </div>
      </section>

      <div className="inspection-layout">
        <InspectionResult result={selected} />
        <DefectHeatmap imageUrl={selected?.heatmap_url} />
      </div>

      <section className="tool-panel">
        <div className="panel-heading">
          <div>
            <h2>Live Inspection Feed</h2>
            <p>{feed.length} recent frames</p>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Source</th>
                <th>Prediction</th>
                <th>Defect</th>
                <th>Severity</th>
                <th>Decision</th>
              </tr>
            </thead>
            <tbody>
              {feed.map((item) => (
                <tr key={item.id}>
                  <td>{formatTime(item.created_at)}</td>
                  <td>{item.source_label}</td>
                  <td>{item.prediction}</td>
                  <td>{item.defect_type}</td>
                  <td>{item.severity_level}</td>
                  <td>{item.pass_fail}</td>
                </tr>
              ))}
              {!feed.length ? (
                <tr>
                  <td colSpan="6">Start the stream to inspect simulated camera frames.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>
    </AppShell>
  );
}
