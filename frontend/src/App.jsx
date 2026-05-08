import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Check,
  CircleDot,
  Dna,
  HeartPulse,
  Play,
  ShieldCheck,
  Stethoscope,
  Zap,
} from 'lucide-react';

const ORCHESTRATION_STEPS = [
  { id: 'cardiology', label: 'Cardiology Agent', icon: HeartPulse, delay: 1000 },
  { id: 'endocrinology', label: 'Endocrinology Agent', icon: Activity, delay: 3000 },
  { id: 'nephrology', label: 'Nephrology Agent', icon: Stethoscope, delay: 5000 },
  { id: 'topsis', label: 'TOPSIS Scoring', icon: Zap, delay: 7000 },
];

const AGENT_URL = import.meta.env.VITE_A2A_AGENT_URL || '';
const A2A_API_KEY = import.meta.env.VITE_A2A_API_KEY || '';
const REQUEST_TIMEOUT_MS = 60000;
const ORCHESTRATION_SUFFIX = 'Run the full multi-specialty orchestration.';

const PATIENTS = {
  patientA: {
    shortName: 'Wei Chen',
    identity: '68M',
    headline: 'HFrEF + CKD Stage 4 + T2DM',
    prompt:
      '68 year old male, HFrEF LVEF 32%, CKD stage 4 eGFR 28, T2DM HbA1c 8.2%, K+ 5.1, on Lisinopril 10mg, Metformin 500mg BID, Furosemide 40mg BID, Aspirin 81mg, Glipizide 5mg BID',
    diagnoses: ['HFrEF LVEF 32%', 'CKD Stage 4 eGFR 28', 'T2DM HbA1c 8.2%'],
    metrics: [
      { label: 'LVEF', value: '32%', tone: 'critical' },
      { label: 'eGFR', value: '28', tone: 'critical' },
      { label: 'HbA1c', value: '8.2%', tone: 'warning' },
      { label: 'K+', value: '5.1', tone: 'warning' },
    ],
    meds: ['Lisinopril 10mg', 'Metformin 500mg BID', 'Furosemide 40mg BID', 'Aspirin 81mg', 'Glipizide 5mg BID'],
    ranking: [
      {
        rank: '🥇',
        specialty: 'Nephrology',
        score: 0.900,
        recommendation: 'STOP metformin immediately (eGFR 28 < 30). Start SGLT2i (dapagliflozin 10mg). Monitor K+ 5.1.',
      },
      {
        rank: '🥈',
        specialty: 'Cardiology',
        score: 0.625,
        recommendation: 'Start carvedilol 3.125mg BID for HFrEF + dapagliflozin 10mg. Continue lisinopril.',
      },
      {
        rank: '🥉',
        specialty: 'Endocrinology',
        score: 0.350,
        recommendation: 'Discontinue metformin (eGFR 28). Start empagliflozin 10mg. Continue glipizide with caution.',
      },
    ],
    conflicts: [
      { icon: Check, tone: 'good', text: 'Stop Metformin — unanimous across all 3 specialties (eGFR <30)' },
      { icon: Check, tone: 'good', text: 'Start SGLT2i — triple benefit for HF + CKD + T2DM' },
      { icon: AlertTriangle, tone: 'warn', text: 'ACEi + CKD Stage 4 — monitor K+ closely (currently 5.1)' },
    ],
  },
  patientB: {
    shortName: 'Maria Santos',
    identity: '55F',
    headline: 'HFpEF without diabetes or CKD',
    prompt:
      '55 year old female, HFpEF LVEF 58%, eGFR 82, no diabetes, on Lisinopril 20mg, Carvedilol 12.5mg BID',
    diagnoses: ['HFpEF LVEF 58%', 'No diabetes', 'No CKD (eGFR 82)'],
    metrics: [
      { label: 'LVEF', value: '58%', tone: 'stable' },
      { label: 'eGFR', value: '82', tone: 'stable' },
      { label: 'HbA1c', value: '5.4%', tone: 'stable' },
      { label: 'K+', value: '4.2', tone: 'stable' },
    ],
    meds: ['Lisinopril 20mg', 'Carvedilol 12.5mg BID'],
    ranking: [
      {
        rank: '🥇',
        specialty: 'Cardiology',
        score: 0.900,
        recommendation: 'Continue carvedilol + lisinopril. Consider SGLT2i for HFpEF symptom management.',
      },
      {
        rank: '🥈',
        specialty: 'Endocrinology',
        score: 0.625,
        recommendation: 'No glucose-lowering therapy needed (HbA1c 5.4%). Continue current HF management.',
      },
      {
        rank: '🥉',
        specialty: 'Nephrology',
        score: 0.350,
        recommendation: 'No nephrotoxic concerns. eGFR 82 normal. Continue lisinopril for HF + HTN.',
      },
    ],
    conflicts: [
      { icon: Check, tone: 'good', text: 'No Metformin — patient is not on Metformin' },
      { icon: Check, tone: 'good', text: 'No CKD or T2DM — fewer cross-specialty conflicts' },
      { icon: Check, tone: 'good', text: 'HFpEF managed with carvedilol + lisinopril' },
    ],
  },
};

function createRequestId() {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function extractAgentText(responseJson) {
  const candidates = [
    responseJson?.result?.artifacts?.[0]?.parts?.[0]?.text,
    responseJson?.result?.status?.message?.parts?.[0]?.text,
    responseJson?.result?.task?.artifacts?.[0]?.parts?.[0]?.text,
    responseJson?.result?.task?.status?.message?.parts?.[0]?.text,
  ];
  return candidates.find((text) => typeof text === 'string' && text.trim())?.trim() || '';
}

function stripMarkdown(value) {
  return value
    .replace(/\*\*/g, '')
    .replace(/`/g, '')
    .trim();
}

function parseRankingRows(text) {
  const rows = [];
  const rankLabels = {
    1: '🥇',
    2: '🥈',
    3: '🥉',
  };
  const rankPattern =
    /^\|\s*(🥇|🥈|🥉|[123])\s*\|\s*([^|]+?)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|\s*([^|]+?)\s*\|?\s*$/gm;
  let match;
  while ((match = rankPattern.exec(text)) !== null) {
    const specialty = stripMarkdown(match[2]);
    if (/specialty/i.test(specialty)) continue;
    rows.push({
      rank: rankLabels[match[1]] || match[1],
      specialty,
      score: Number.parseFloat(match[3]),
      recommendation: stripMarkdown(match[4]),
    });
  }
  return rows.filter((row) => row.specialty && Number.isFinite(row.score) && row.recommendation);
}

function parseTopPick(text, ranking) {
  const match = text.match(/\*\*Top Pick:\*\*\s*([^-—\n]+?)\s*[-—]\s*([^\n]+)/i);
  if (match) {
    return {
      specialty: stripMarkdown(match[1]),
      recommendation: stripMarkdown(match[2]),
    };
  }
  if (ranking[0]) {
    return {
      specialty: ranking[0].specialty,
      recommendation: ranking[0].recommendation,
    };
  }
  return null;
}

function parseConflicts(text) {
  const section = text.match(
    /###\s*Key Conflicts Resolved\s*\n([\s\S]*?)(?=\n###|\n\*\*Citations|\n\*\*Disclaimer|\n---|$)/i,
  );
  if (!section) return [];
  return section[1]
    .split('\n')
    .map((line) => line.match(/^\s*[-*]\s+(.+?)\s*$/)?.[1])
    .filter(Boolean)
    .map((item) => ({
      icon: Check,
      tone: /risk|withhold|contraindicat|stop|safety|threshold/i.test(item) ? 'warn' : 'good',
      text: stripMarkdown(item),
    }));
}

function parseAgentText(text) {
  const ranking = parseRankingRows(text);
  const conflicts = parseConflicts(text);
  const topPick = parseTopPick(text, ranking);

  if (!ranking.length || !conflicts.length) {
    return {
      rawText: text,
      source: 'raw',
    };
  }

  return {
    ranking,
    conflicts,
    topPick,
    rawText: text,
    source: 'agent',
  };
}

async function sendA2ARequest(patientInput) {
  if (!AGENT_URL) {
    throw new Error('VITE_A2A_AGENT_URL is not configured');
  }

  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const jsonRpcId = createRequestId();
  const messageId = createRequestId();

  try {
    const headers = {
      'Content-Type': 'application/json',
    };
    if (A2A_API_KEY) {
      headers['X-API-Key'] = A2A_API_KEY;
    }

    const response = await fetch(AGENT_URL, {
      method: 'POST',
      headers,
      signal: controller.signal,
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'message/send',
        params: {
          message: {
            role: 'user',
            parts: [{ kind: 'text', text: `${patientInput.trim()} ${ORCHESTRATION_SUFFIX}` }],
            messageId,
          },
        },
        id: jsonRpcId,
      }),
    });

    if (!response.ok) {
      throw new Error(`Agent request failed with ${response.status}`);
    }

    const responseJson = await response.json();
    const text = extractAgentText(responseJson);
    if (!text) {
      throw new Error('Agent response did not include text output');
    }
    return parseAgentText(text);
  } finally {
    window.clearTimeout(timeout);
  }
}

function App() {
  const [selectedId, setSelectedId] = useState('patientA');
  const [isRunning, setIsRunning] = useState(false);
  const [isWaitingForBackend, setIsWaitingForBackend] = useState(false);
  const [completedSteps, setCompletedSteps] = useState([]);
  const [activeStep, setActiveStep] = useState(null);
  const [hasRun, setHasRun] = useState(false);
  const [patientInput, setPatientInput] = useState(PATIENTS.patientA.prompt);
  const [liveResult, setLiveResult] = useState(null);
  const [errorBanner, setErrorBanner] = useState(
    !AGENT_URL ? 'VITE_A2A_AGENT_URL is not configured' : A2A_API_KEY ? '' : 'VITE_A2A_API_KEY is not configured',
  );
  const timersRef = useRef([]);

  const selectedPatient = PATIENTS[selectedId];
  const displayRanking = liveResult?.ranking || selectedPatient.ranking;
  const displayConflicts = liveResult?.conflicts || selectedPatient.conflicts;
  const displayTopPick = liveResult?.topPick || null;
  const rawAgentText = liveResult?.source === 'raw' ? liveResult.rawText : '';
  const usingDemoData = Boolean(errorBanner && hasRun && !liveResult?.ranking && !rawAgentText);

  const progressPercent = useMemo(
    () => Math.round((completedSteps.length / ORCHESTRATION_STEPS.length) * 100),
    [completedSteps],
  );

  useEffect(() => {
    return () => timersRef.current.forEach(clearTimeout);
  }, []);

  function resetRunState() {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
    setIsRunning(false);
    setIsWaitingForBackend(false);
    setCompletedSteps([]);
    setActiveStep(null);
    setHasRun(false);
    setLiveResult(null);
    setErrorBanner(!AGENT_URL ? 'VITE_A2A_AGENT_URL is not configured' : A2A_API_KEY ? '' : 'VITE_A2A_API_KEY is not configured');
  }

  function handlePatientChange(patientId) {
    if (isRunning) return;
    if (patientId === selectedId) return;
    resetRunState();
    setSelectedId(patientId);
    setPatientInput(PATIENTS[patientId].prompt);
  }

  function runStatusAnimation() {
    return new Promise((resolve) => {
      ORCHESTRATION_STEPS.forEach((step, index) => {
        const timer = setTimeout(() => {
          setCompletedSteps((current) => {
            if (current.includes(step.id)) return current;
            return [...current, step.id];
          });
          const next = ORCHESTRATION_STEPS[index + 1];
          setActiveStep(next?.id ?? null);
          if (!next) resolve();
        }, step.delay);
        timersRef.current.push(timer);
      });
    });
  }

  async function runOrchestration() {
    if (isRunning) return;
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
    setHasRun(false);
    setIsRunning(true);
    setIsWaitingForBackend(true);
    setCompletedSteps([]);
    setActiveStep(ORCHESTRATION_STEPS[0].id);
    setLiveResult(null);
    setErrorBanner(!AGENT_URL ? 'VITE_A2A_AGENT_URL is not configured' : A2A_API_KEY ? '' : 'VITE_A2A_API_KEY is not configured');

    const animationPromise = runStatusAnimation();
    try {
      const result = await sendA2ARequest(patientInput);
      await animationPromise;
      setLiveResult(result);
    } catch (error) {
      console.error(error);
      await animationPromise;
      setErrorBanner('Agent unavailable, showing demo data');
      setLiveResult(null);
    } finally {
      timersRef.current.forEach(clearTimeout);
      timersRef.current = [];
      setCompletedSteps(ORCHESTRATION_STEPS.map((step) => step.id));
      setActiveStep(null);
      setIsWaitingForBackend(false);
      setIsRunning(false);
      setHasRun(true);
    }
  }

  return (
    <main className="app-shell">
      <aside className="side-rail" aria-label="Console controls">
        <div className="brand-block">
          <div className="brand-mark">
            <Dna size={24} />
          </div>
          <div>
            <strong>Consilium</strong>
            <span>Clinical AI</span>
          </div>
        </div>

        <div className="rail-section">
          <span className="section-kicker">Patient presets</span>
          <div className="patient-controls" aria-label="Patient presets">
            {Object.entries(PATIENTS).map(([id, preset]) => (
              <button
                key={id}
                className={id === selectedId ? 'preset-button active' : 'preset-button'}
                onClick={() => handlePatientChange(id)}
              >
                {preset.shortName}
              </button>
            ))}
          </div>
        </div>

        <PatientInput value={patientInput} onChange={setPatientInput} disabled={isRunning} />

        <button className="run-button" onClick={runOrchestration} disabled={isRunning}>
          <Play size={17} fill="currentColor" />
          {isWaitingForBackend ? 'Waiting for Agent...' : isRunning ? 'Orchestrating...' : 'Run Orchestration'}
        </button>

        <RunMeter progressPercent={progressPercent} isRunning={isRunning} hasRun={hasRun} />
      </aside>

      <section className="console-shell">
        <div className="top-region">
          <section className="topbar" aria-label="Application header">
            <div>
              <div className="eyebrow">
                <span className="pulse-dot" />
                Multi-agent decision console
              </div>
              <h1>Consilium — Multi-Specialty Clinical Decision System</h1>
            </div>
            <div className={AGENT_URL && A2A_API_KEY ? 'system-badge' : 'system-badge warning'}>
              <ShieldCheck size={17} />
              {AGENT_URL && A2A_API_KEY ? 'A2A Agent ready' : 'Agent config needed'}
            </div>
          </section>

          {errorBanner && <div className="error-banner">{errorBanner}</div>}
        </div>

        <section className="clinical-grid" aria-label="Clinical workspace">
          <PatientHeader patient={selectedPatient} />
          <AgentStatusPanel activeStep={activeStep} completedSteps={completedSteps} />
          <DecisionPanel
            ranking={displayRanking}
            topPick={displayTopPick}
            rawText={rawAgentText}
            hasRun={hasRun}
            isRunning={isRunning}
            usingDemoData={usingDemoData}
          />
        </section>

        <ConflictPanel conflicts={displayConflicts} hasRun={hasRun} isRunning={isRunning} />
      </section>
    </main>
  );
}

function RunMeter({ progressPercent, isRunning, hasRun }) {
  return (
    <div className="run-meter" aria-label="Orchestration progress">
      <div>
        <span>{isRunning ? 'Running' : hasRun ? 'Complete' : 'Ready'}</span>
        <strong>{progressPercent}%</strong>
      </div>
      <div className="meter-track">
        <div className="meter-fill" style={{ width: `${progressPercent}%` }} />
      </div>
    </div>
  );
}

function PatientInput({ value, onChange, disabled }) {
  return (
    <label className="patient-input-block">
      <span className="strip-label">Patient input</span>
      <textarea
        value={value}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
        rows={6}
        placeholder="Describe the patient summary for orchestration"
      />
    </label>
  );
}

function PatientHeader({ patient }) {
  return (
    <section className="patient-card" aria-label="Patient information">
      <div className="patient-summary">
        <span className="section-kicker">Patient profile</span>
        <div className="patient-id-row">
          <span className="patient-id">{patient.identity}</span>
          <span>{patient.headline}</span>
        </div>
        <div className="diagnosis-row">
          {patient.diagnoses.map((diagnosis) => (
            <span key={diagnosis}>{diagnosis}</span>
          ))}
        </div>
      </div>

      <div className="metric-strip" aria-label="Key metrics">
        {patient.metrics.map((metric) => (
          <div className={`metric-cell ${metric.tone}`} key={metric.label}>
            <span>{metric.label}</span>
            <strong>{metric.value}</strong>
          </div>
        ))}
      </div>

      <div className="medication-strip">
        <span className="strip-label">Current meds</span>
        <div className="med-list">
          {patient.meds.map((med) => (
            <span key={med}>{med}</span>
          ))}
        </div>
      </div>

    </section>
  );
}

function AgentStatusPanel({ activeStep, completedSteps }) {
  return (
    <section className="panel status-panel" aria-label="Agent status panel">
      <div className="panel-heading">
        <div>
          <span className="section-kicker">Agent status</span>
          <h2>Specialist execution</h2>
        </div>
        <Activity size={22} />
      </div>

      <div className="status-list">
        {ORCHESTRATION_STEPS.map((step, index) => {
          const complete = completedSteps.includes(step.id);
          const active = activeStep === step.id;
          const pending = !complete && !active;
          const StepIcon = step.icon;
          return (
            <div
              className={`status-row ${complete ? 'complete' : ''} ${active ? 'active' : ''} ${pending ? 'pending' : ''}`}
              key={step.id}
              style={{ '--row-index': index }}
            >
              <div className="step-icon" aria-hidden="true">
                <StepIcon size={18} />
              </div>
              <div className="step-copy">
                <strong>{step.label}</strong>
                <span>{complete ? 'Complete' : active ? 'Analyzing...' : 'Waiting'}</span>
              </div>
              <div className="step-state">
                {complete ? <Check size={18} /> : active ? <CircleDot size={18} /> : <span />}
              </div>
            </div>
          );
        })}
      </div>

      <div className="panel-footer">
        <ShieldCheck size={18} />
        <span>FHIR context isolated from model prompts</span>
      </div>
    </section>
  );
}

function DecisionPanel({ ranking, topPick, rawText, hasRun, isRunning, usingDemoData }) {
  return (
    <section className={hasRun ? 'decision-panel has-results' : 'decision-panel'} aria-label="TOPSIS ranking results">
      <div className="body-visual" aria-hidden="true">
        <div className="clinical-plus-visual">
          <span className="circuit-line line-a" />
          <span className="circuit-line line-b" />
          <span className="circuit-line line-c" />
          <div className="plus-stack stack-back" />
          <div className="plus-stack stack-mid" />
          <div className="plus-stack stack-front">
            <span className="plus-cross" />
          </div>
        </div>
      </div>

      <div className="decision-content">
        <div className="panel-heading">
          <div>
            <span className="section-kicker">TOPSIS ranking</span>
            <h2>Decision priority</h2>
          </div>
          <Dna size={22} />
        </div>

        {!hasRun && (
          <div className="locked-state">
            <HeartPulse size={28} />
            <strong>{isRunning ? 'Specialists still analyzing' : 'Awaiting orchestration'}</strong>
            <span>Priority ranking appears after all specialist agents and TOPSIS scoring complete.</span>
          </div>
        )}

        {hasRun && (
          rawText ? (
            <div className="raw-response">
              <strong>Raw Agent Response</strong>
              <pre>{rawText}</pre>
            </div>
          ) : (
            <div className="ranking-list visible">
              {usingDemoData && <div className="result-source">Demo fallback data</div>}
              {topPick && (
                <div className="top-pick">
                  <span>Top Pick</span>
                  <div>
                    <strong>{topPick.specialty}</strong>
                    <small>{topPick.recommendation}</small>
                  </div>
                </div>
              )}
              {ranking.map((item, index) => (
                <article className="ranking-row" key={item.specialty} style={{ '--rank-index': index }}>
                  <div className="rank-main">
                    <span className="rank-medal">{item.rank}</span>
                    <div>
                      <h3>{item.specialty}</h3>
                      <p>{item.recommendation}</p>
                    </div>
                    <strong className="rank-score">{item.score.toFixed(3)}</strong>
                  </div>
                  <div className="score-track">
                    <div className="score-fill" style={{ width: `${Math.min(item.score, 1) * 100}%` }} />
                  </div>
                </article>
              ))}
            </div>
          )
        )}
      </div>
    </section>
  );
}

function ConflictPanel({ conflicts, hasRun, isRunning }) {
  return (
    <section className={hasRun ? 'conflict-panel resolved' : 'conflict-panel'} aria-label="Resolved conflicts">
      <div className="conflict-heading">
        <div>
          <span className="section-kicker">Key conflicts resolved</span>
          <h2>Unified clinical action set</h2>
        </div>
        <span className="heading-action">
          <Zap size={21} />
        </span>
      </div>

      {!hasRun && (
        <div className="consensus-waiting">
          <CircleDot size={18} />
          <span>
            {isRunning
              ? 'Waiting for all agents to finish before publishing the unified action set.'
              : 'Run orchestration to generate the cross-specialty consensus.'}
          </span>
        </div>
      )}

      {hasRun && (
        <div className="conflict-grid">
          {conflicts.map((conflict, index) => {
            const Icon = conflict.icon;
            return (
              <div className={`conflict-item ${conflict.tone}`} key={conflict.text} style={{ '--conflict-index': index }}>
                <Icon size={20} />
                <span>{conflict.text}</span>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

export default App;
