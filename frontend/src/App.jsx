import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Check,
  CircleDot,
  Dna,
  LockKeyhole,
  HeartPulse,
  Play,
  ShieldCheck,
  Stethoscope,
  Trash2,
  Upload,
  Zap,
} from 'lucide-react';

const PIPELINE_STEPS = [
  { id: 'fhir', label: 'FHIR Data Retrieval', icon: Activity, delay: 2000 },
  { id: 'cardiology', label: 'Cardiology Agent', icon: HeartPulse, delay: 4000 },
  { id: 'nephrology', label: 'Nephrology Agent', icon: Stethoscope, delay: 6000 },
  { id: 'endocrinology', label: 'Endocrinology Agent', icon: Activity, delay: 8000 },
  { id: 'topsis', label: 'TOPSIS Scoring', icon: Zap, delay: 10000 },
  { id: 'format', label: 'Formatting Output', icon: ShieldCheck },
];

const AGENT_URL = import.meta.env.VITE_A2A_AGENT_URL || '';
const A2A_API_KEY = import.meta.env.VITE_A2A_API_KEY || '';
const REQUEST_TIMEOUT_MS = 60000;
const ORCHESTRATION_SUFFIX = 'Run the full multi-specialty orchestration.';
const TIMED_PIPELINE_STEPS = PIPELINE_STEPS.filter((step) => Number.isFinite(step.delay));
const DEFAULT_TRACE = [
  { specialty: 'Cardiology', text: 'Specialist recommendation captured' },
  { specialty: 'Nephrology', text: 'Renal safety reviewed' },
  { specialty: 'Endocrinology', text: 'Metabolic therapy reviewed' },
  { specialty: 'Consensus', text: 'Deterministic clinical ranking applied', consensus: true },
];

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
    trace: [
      { specialty: 'Cardiology', text: 'Preserve HF mortality benefit' },
      { specialty: 'Nephrology', text: 'eGFR 28 triggers renal safety override' },
      { specialty: 'Endocrinology', text: 'Replace Metformin with SGLT2i' },
      { specialty: 'Consensus', text: 'Stop Metformin first; align on SGLT2i', consensus: true },
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
    trace: [
      { specialty: 'Cardiology', text: 'HFpEF remains the active priority' },
      { specialty: 'Nephrology', text: 'No CKD safety blocker detected' },
      { specialty: 'Endocrinology', text: 'No diabetes therapy required' },
      { specialty: 'Consensus', text: 'Continue HF management; no false CKD/DM conflict', consensus: true },
    ],
  },
};

/** Parse a FHIR R4 transaction Bundle into the frontend patient format. */
function parseFhirBundle(bundle) {
  const entries = bundle?.entry || [];
  const get = (type) =>
    entries
      .filter((e) => e.resource?.resourceType === type)
      .map((e) => e.resource);

  // Patient demographics
  const patient = get('Patient')[0];
  if (!patient) throw new Error('No Patient resource found in bundle');
  const nameObj = patient.name?.[0] || {};
  const given = (nameObj.given || []).join(' ');
  const family = nameObj.family || '';
  const fullName = `${given} ${family}`.trim() || 'Imported Patient';
  const birthDate = patient.birthDate || '';
  const gender = patient.gender || 'unknown';
  const age = birthDate
    ? Math.floor((Date.now() - new Date(birthDate).getTime()) / (365.25 * 864e5))
    : null;
  const identity = age !== null ? `${age}${gender === 'female' ? 'F' : 'M'}` : gender;

  // Conditions
  const conditions = get('Condition');
  const diagnosisNames = conditions.map(
    (c) => c.code?.text || c.code?.coding?.[0]?.display || 'Unknown condition',
  );

  // Observations — extract key metrics
  const observations = get('Observation');
  function findObs(...loincCodes) {
    return observations.find((o) =>
      o.code?.coding?.some((c) => loincCodes.includes(c.code)),
    );
  }
  const efObs = findObs('10230-1');
  const egfrObs = findObs('48642-3');
  const hba1cObs = findObs('4548-4');
  const kObs = findObs('2823-3');
  const bnpObs = findObs('30934-4');
  const crObs = findObs('2160-0');

  function val(obs) {
    if (!obs?.valueQuantity) return null;
    return { value: String(obs.valueQuantity.value), unit: obs.valueQuantity.unit || '' };
  }
  const efVal = val(efObs);
  const egfrVal = val(egfrObs);
  const hba1cVal = val(hba1cObs);
  const kVal = val(kObs);

  // Tone helper
  function metricTone(label, value) {
    const n = parseFloat(value);
    if (Number.isNaN(n)) return 'stable';
    if (label === 'LVEF' && n < 40) return 'critical';
    if (label === 'eGFR' && n < 30) return 'critical';
    if (label === 'eGFR' && n < 60) return 'warning';
    if (label === 'HbA1c' && n > 7.5) return 'warning';
    if (label === 'K+' && n > 5.0) return 'warning';
    return 'stable';
  }

  const metrics = [];
  if (efVal) metrics.push({ label: 'LVEF', value: `${efVal.value}%`, tone: metricTone('LVEF', efVal.value) });
  if (egfrVal) metrics.push({ label: 'eGFR', value: egfrVal.value, tone: metricTone('eGFR', egfrVal.value) });
  if (hba1cVal) metrics.push({ label: 'HbA1c', value: `${hba1cVal.value}%`, tone: metricTone('HbA1c', hba1cVal.value) });
  if (kVal) metrics.push({ label: 'K+', value: kVal.value, tone: metricTone('K+', kVal.value) });

  // Medications
  const medRequests = get('MedicationRequest');
  const meds = medRequests.map(
    (m) =>
      m.medicationCodeableConcept?.text ||
      m.medicationCodeableConcept?.coding?.[0]?.display ||
      'Unknown medication',
  );

  // Build headline from conditions
  const headlineParts = [];
  if (efObs) {
    const ef = parseFloat(efVal?.value);
    headlineParts.push(ef < 40 ? 'HFrEF' : ef >= 50 ? 'HFpEF' : 'HFmrEF');
  }
  if (egfrVal) {
    const egfr = parseFloat(egfrVal.value);
    if (egfr < 15) headlineParts.push('CKD Stage 5');
    else if (egfr < 30) headlineParts.push('CKD Stage 4');
    else if (egfr < 60) headlineParts.push('CKD Stage 3');
  }
  if (hba1cVal && parseFloat(hba1cVal.value) >= 6.5) headlineParts.push('T2DM');

  // Build prompt for A2A
  const promptParts = [`${age || ''} year old ${gender}`];
  if (efVal) promptParts.push(`LVEF ${efVal.value}%`);
  if (egfrVal) promptParts.push(`eGFR ${egfrVal.value}`);
  if (hba1cVal) promptParts.push(`HbA1c ${hba1cVal.value}%`);
  if (kVal) promptParts.push(`K+ ${kVal.value}`);
  if (meds.length) promptParts.push(`on ${meds.join(', ')}`);

  return {
    shortName: fullName,
    identity,
    headline: headlineParts.length ? headlineParts.join(' + ') : 'Imported patient',
    prompt: promptParts.join(', '),
    diagnoses: diagnosisNames.length ? diagnosisNames : ['See FHIR data'],
    metrics,
    meds: meds.length ? meds : ['None recorded'],
    ranking: [],
    conflicts: [],
    trace: DEFAULT_TRACE,
    _imported: true,
  };
}

function createRequestId() {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function createImportedPatientId() {
  return `imported-${createRequestId()}`;
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
  const [hasRun, setHasRun] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [completedStepIds, setCompletedStepIds] = useState([]);
  const [importedPatients, setImportedPatients] = useState([]);
  const [liveResult, setLiveResult] = useState(null);
  const [errorBanner, setErrorBanner] = useState(
    !AGENT_URL ? 'VITE_A2A_AGENT_URL is not configured' : A2A_API_KEY ? '' : 'VITE_A2A_API_KEY is not configured',
  );
  const timersRef = useRef([]);
  const fileInputRef = useRef(null);
  const tickRef = useRef(null);

  const selectedPatient = useMemo(
    () =>
      importedPatients.find((patient) => patient.id === selectedId) ||
      PATIENTS[selectedId] ||
      PATIENTS.patientA,
    [importedPatients, selectedId],
  );
  const patientPrompt = selectedPatient.prompt;
  const displayRanking = liveResult?.ranking || selectedPatient.ranking;
  const displayConflicts = liveResult?.conflicts || selectedPatient.conflicts;
  const displayTrace = selectedPatient.trace || DEFAULT_TRACE;
  const displayTopPick = liveResult?.topPick || null;
  const rawAgentText = liveResult?.source === 'raw' ? liveResult.rawText : '';
  const usingDemoData = Boolean(errorBanner && hasRun && !liveResult?.ranking && !rawAgentText);


  useEffect(() => {
    return () => {
      timersRef.current.forEach(window.clearTimeout);
      window.clearInterval(tickRef.current);
    };
  }, []);

  function resetRunState() {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
    window.clearInterval(tickRef.current);
    setIsRunning(false);
    setElapsed(0);
    setCompletedStepIds([]);
    setHasRun(false);
    setLiveResult(null);
    setErrorBanner(!AGENT_URL ? 'VITE_A2A_AGENT_URL is not configured' : A2A_API_KEY ? '' : 'VITE_A2A_API_KEY is not configured');
  }

  function handlePatientChange(patientId) {
    if (isRunning) return;
    if (patientId === selectedId) return;
    resetRunState();
    setSelectedId(patientId);
  }

  function handleFileImport(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const bundle = JSON.parse(e.target.result);
        const parsed = parseFhirBundle(bundle);
        const importedId = createImportedPatientId();
        resetRunState();
        setImportedPatients((current) => [...current, { ...parsed, id: importedId }]);
        setSelectedId(importedId);
      } catch (err) {
        setErrorBanner(`Import failed: ${err.message}`);
      }
    };
    reader.readAsText(file);
    event.target.value = '';
  }

  function handleImportedPatientDelete(patientId) {
    if (isRunning) return;
    resetRunState();
    setImportedPatients((current) => current.filter((patient) => patient.id !== patientId));
    if (selectedId === patientId) {
      setSelectedId('patientA');
    }
  }

  async function runOrchestration() {
    if (isRunning) return;
    setHasRun(false);
    setIsRunning(true);
    setElapsed(0);
    setCompletedStepIds([]);
    setLiveResult(null);
    setErrorBanner(!AGENT_URL ? 'VITE_A2A_AGENT_URL is not configured' : A2A_API_KEY ? '' : 'VITE_A2A_API_KEY is not configured');

    const startTime = Date.now();
    tickRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTime) / 1000));
    }, 200);
    timersRef.current = TIMED_PIPELINE_STEPS.map((step) =>
      window.setTimeout(() => {
        setCompletedStepIds((current) =>
          current.includes(step.id) ? current : [...current, step.id],
        );
      }, step.delay),
    );

    try {
      const result = await sendA2ARequest(patientPrompt);
      setLiveResult(result);
    } catch (error) {
      console.error(error);
      setErrorBanner('Agent unavailable, showing demo data');
      setLiveResult(null);
    } finally {
      window.clearInterval(tickRef.current);
      setElapsed(Math.floor((Date.now() - startTime) / 1000));
      timersRef.current.forEach(clearTimeout);
      timersRef.current = [];
      setCompletedStepIds(PIPELINE_STEPS.map((step) => step.id));
      setIsRunning(false);
      setHasRun(true);
    }
  }

  return (
    <main className={hasRun ? 'app-shell results-mode' : 'app-shell'}>
      <aside className="side-rail" aria-label="Console controls">
        <div className="brand-block">
          <img className="brand-mark" src="/consilium-icon.png" alt="" />
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
            {importedPatients.map((patient) => (
              <div className="imported-patient-row" key={patient.id}>
                <button
                  className={patient.id === selectedId ? 'preset-button imported active' : 'preset-button imported'}
                  onClick={() => handlePatientChange(patient.id)}
                >
                  {patient.shortName}
                </button>
                <button
                  className="delete-patient-button"
                  onClick={(event) => {
                    event.stopPropagation();
                    handleImportedPatientDelete(patient.id);
                  }}
                  disabled={isRunning}
                  aria-label={`Delete ${patient.shortName}`}
                  title={`Delete ${patient.shortName}`}
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="action-block">
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            style={{ display: 'none' }}
            onChange={handleFileImport}
          />
          <button
            className="import-button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isRunning}
          >
            <Upload size={16} />
            Import FHIR Bundle
          </button>

          <button className="run-button" onClick={runOrchestration} disabled={isRunning}>
            <Play size={17} fill="currentColor" />
            {isRunning ? 'Orchestrating...' : 'Run Orchestration'}
          </button>
        </div>

        <RunMeter elapsed={elapsed} isRunning={isRunning} hasRun={hasRun} />
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
          <AgentStatusPanel isRunning={isRunning} hasRun={hasRun} completedStepIds={completedStepIds} />
          <DecisionPanel
            ranking={displayRanking}
            topPick={displayTopPick}
            rawText={rawAgentText}
            hasRun={hasRun}
            isRunning={isRunning}
            usingDemoData={usingDemoData}
          />
        </section>

        <ConflictPanel conflicts={displayConflicts} trace={displayTrace} hasRun={hasRun} isRunning={isRunning} />
      </section>
    </main>
  );
}

function RunMeter({ elapsed, isRunning, hasRun }) {
  const display = isRunning
    ? `${elapsed}s`
    : hasRun
      ? `${elapsed}s`
      : 'Ready';
  return (
    <div className={`run-meter ${isRunning ? 'active' : ''}`} aria-label="Orchestration progress">
      <div>
        <span>{isRunning ? 'Running' : hasRun ? 'Complete' : 'Ready'}</span>
        <strong>{display}</strong>
      </div>
      <div className="meter-track">
        {isRunning && <div className="meter-fill indeterminate" />}
        {hasRun && <div className="meter-fill complete" />}
      </div>
    </div>
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

function AgentStatusPanel({ isRunning, hasRun, completedStepIds }) {
  const completedSet = new Set(completedStepIds);
  const activeIndex = isRunning
    ? PIPELINE_STEPS.findIndex((step) => !completedSet.has(step.id))
    : -1;
  return (
    <section className="panel status-panel" aria-label="Agent status panel">
      <div className="panel-heading">
        <div>
          <span className="section-kicker">Agent status</span>
          <h2>Pipeline</h2>
        </div>
        <Activity size={22} />
      </div>

      <div className="status-list">
        {PIPELINE_STEPS.map((step, index) => {
          const StepIcon = step.icon;
          const complete = completedSet.has(step.id) || (hasRun && !isRunning);
          const active = isRunning && index === activeIndex;
          const pending = !complete && !active;
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
                <span>{complete ? 'Done' : active ? 'Processing...' : 'Waiting'}</span>
              </div>
              <div className="step-state">
                {complete ? <Check size={18} /> : active ? <CircleDot size={18} /> : <span />}
              </div>
            </div>
          );
        })}
      </div>

      <div className="panel-footer">
        <LockKeyhole size={16} />
        <span>FHIR context isolated from model prompts</span>
      </div>
    </section>
  );
}

function DecisionPanel({ ranking, topPick, rawText, hasRun, isRunning, usingDemoData }) {
  return (
    <section className={hasRun ? 'decision-panel has-results' : 'decision-panel'} aria-label="TOPSIS ranking results">
      <div className={`body-visual ${hasRun ? 'fade-out' : ''}`} aria-hidden="true">
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

      <div className={`decision-content ${hasRun ? 'results-visible' : ''}`}>
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

function ConflictPanel({ conflicts, trace, hasRun, isRunning }) {
  return (
    <section className={hasRun ? 'conflict-panel resolved' : 'conflict-panel'} aria-label="Resolved conflicts">
      <div className="conflict-heading">
        <div>
          <span className="section-kicker">Key conflicts resolved</span>
          <h2>Unified clinical action set</h2>
        </div>
      </div>

      {!hasRun && (
        <div className="consensus-waiting">
          <CircleDot size={18} />
          <div>
            <span>
              {isRunning
                ? 'Waiting for all agents to finish before publishing the unified action set.'
                : 'Run orchestration to generate the cross-specialty consensus.'}
            </span>
            <small>Run orchestration to reveal the reconciliation trace.</small>
          </div>
        </div>
      )}

      {hasRun && (
        <div className="conflict-results">
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

          <div className="trace-panel" aria-label="Clinical conflict trace">
            <span className="trace-kicker">Clinical conflict trace</span>
            <div className="trace-list">
              {trace.map((item, index) => (
                <div
                  className={item.consensus ? 'trace-item consensus' : 'trace-item'}
                  key={`${item.specialty}-${item.text}`}
                  style={{ '--trace-index': index }}
                >
                  <strong>{item.specialty}</strong>
                  <span>{item.text}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

export default App;
