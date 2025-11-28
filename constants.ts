import { AgentState, AgentStatus } from "./types";

// Simulated "Long-term Memory" / Knowledge Base
export const KNOWLEDGE_BASE_SOPS = `
SOP-101: Temperature Control in Curing Process
- Optimal range: 145°C - 155°C.
- Deviation > 5°C for > 10 mins results in "brittle fracture" risk.
- Action: Check thermocouple calibration and PID controller gains.

SOP-204: Pressure Regulation
- Target pressure: 4500 PSI.
- Fluctuations > 10% indicate hydraulic leak or pump cavitation.
- Correlated with "surface pitting" defects.

SOP-305: Shift Handover Protocol
- Operators must log all "Yield Excursions" manually.
- Sudden drops in yield often correlate with Shift A to Shift B transition due to machine recalibration habits.
- Action: Standardize calibration offset to 0.0 before shift end.

Historical Case #882:
- Issue: Batch 45-50 showed high rejection rate.
- Cause: Raw material lot #992 was contaminated with moisture.
- Signature: Spike in humidity sensor coupled with lower viscosity.
`;

export const INITIAL_AGENTS: AgentState[] = [
  {
    id: 'data-agent',
    name: 'DataAgent',
    role: 'The Analyst',
    status: AgentStatus.IDLE,
    message: 'Waiting for data...'
  },
  {
    id: 'insight-agent',
    name: 'InsightAgent',
    role: 'The Expert',
    status: AgentStatus.IDLE,
    message: 'Waiting for analysis...'
  },
  {
    id: 'knowledge-agent',
    name: 'KnowledgeAgent',
    role: 'The Librarian',
    status: AgentStatus.IDLE,
    message: 'Waiting for insights...'
  },
  {
    id: 'report-agent',
    name: 'ReportAgent',
    role: 'The Communicator',
    status: AgentStatus.IDLE,
    message: 'Waiting for synthesis...'
  }
];

export const SAMPLE_CSV = `BatchID,Temperature,Pressure,Humidity,Yield,DefectRate
1,150,4500,45,98.5,1.5
2,151,4510,46,98.2,1.8
3,149,4490,44,99.0,1.0
4,152,4520,45,97.8,2.2
5,155,4600,48,96.5,3.5
6,158,4650,50,92.0,8.0
7,160,4700,52,85.0,15.0
8,162,4750,55,80.0,20.0
9,150,4500,45,98.0,2.0
10,149,4495,44,98.8,1.2`;