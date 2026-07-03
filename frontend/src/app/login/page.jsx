"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Factory, LockKeyhole, UserPlus } from "lucide-react";

import { login, registerUser } from "../../services/authApi";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({
    name: "Quality Engineer",
    email: "admin@visioninspect.ai",
    password: "Admin@12345",
    role: "quality_engineer",
  });
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);

  function updateField(field, value) {
    setForm((current) => ({ ...current, [field]: value }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setSuccess("");
    setLoading(true);

    try {
      if (mode === "login") {
        await login({ email: form.email, password: form.password });
        router.replace("/dashboard");
      } else {
        const result = await registerUser(form);
        setSuccess(result.message || "Registration request submitted for admin approval.");
        setMode("login");
      }
    } catch (err) {
      setError(err.message || "Authentication failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="auth-screen">
      <section className="auth-panel">
        <div className="auth-brand">
          <span className="brand-mark">
            <Factory size={24} />
          </span>
          <div>
            <h1>VisionInspect AI</h1>
            <p>Manufacturing defect detection console</p>
          </div>
        </div>

        <div className="segmented-control" role="tablist" aria-label="Authentication mode">
          <button className={mode === "login" ? "active" : ""} type="button" onClick={() => setMode("login")}>
            <LockKeyhole size={16} />
            Login
          </button>
          <button className={mode === "register" ? "active" : ""} type="button" onClick={() => setMode("register")}>
            <UserPlus size={16} />
            Register
          </button>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          {mode === "register" ? (
            <>
              <label>
                Name
                <input value={form.name} onChange={(event) => updateField("name", event.target.value)} />
              </label>
              <label>
                Requested Role
                <select value={form.role} onChange={(event) => updateField("role", event.target.value)}>
                  <option value="quality_engineer">Quality engineer</option>
                  <option value="factory_supervisor">Factory supervisor</option>
                  <option value="quality_manager">Quality manager</option>
                </select>
              </label>
            </>
          ) : null}
          <label>
            Email
            <input type="email" value={form.email} onChange={(event) => updateField("email", event.target.value)} />
          </label>
          <label>
            Password
            <input
              type="password"
              value={form.password}
              onChange={(event) => updateField("password", event.target.value)}
            />
          </label>
          {error ? <p className="form-error">{error}</p> : null}
          {success ? <p className="form-success">{success}</p> : null}
          <button className="primary-button" type="submit" disabled={loading}>
            {mode === "login" ? <LockKeyhole size={18} /> : <UserPlus size={18} />}
            <span>{loading ? "Please wait" : mode === "login" ? "Login" : "Request Account"}</span>
          </button>
        </form>
      </section>
    </main>
  );
}
