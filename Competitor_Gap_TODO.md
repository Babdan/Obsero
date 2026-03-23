# Obsero Competitor Gap TODO List

Purpose: Convert competitor feature gaps into an actionable product roadmap with both technical and commercial context.

Scoring guide:
- Necessity: 1 (nice to have) to 5 (critical)
- Sellability: 1 (hard to market) to 5 (strong buying trigger)
- Build complexity: Low, Medium, High

## P0 - Immediate Priority (High Impact + High Commercial Value)

### [ ] 1. Incident Lifecycle Engine
- What it is:
  A full state machine for incidents (new -> triage -> assigned -> investigating -> confirmed/false positive -> closed).
- What it does:
  Standardizes handling, ownership, and closure steps for every incident.
- How it helps:
  Reduces operational chaos, improves accountability, and supports audits/compliance.
- What we have that is similar:
  Alert statuses and labeling endpoints exist; no full case workflow with assignment/transition controls.
- Gap status:
  Partially present.
- Necessity: 5/5
- Sellability: 5/5
- Build complexity: Medium
- Why customers care:
  This is often a procurement requirement for EHS buyers.

### [ ] 2. SLA Timers + Auto-Escalation
- What it is:
  Rule-based timers for triage/ack/close with escalation to supervisor/manager when breached.
- What it does:
  Prevents incidents from being ignored and enforces response discipline.
- How it helps:
  Improves MTTA/MTTR and trust in operations.
- What we have that is similar:
  Cooldowns and temporal gating for detections; no case-level SLA engine.
- Gap status:
  Completely missing.
- Necessity: 5/5
- Sellability: 5/5
- Build complexity: Medium
- Why customers care:
  Highly visible KPI and easy value story for management.

### [ ] 3. Corrective Actions Module
- What it is:
  Turn incidents/findings into tasks with owner, due date, progress, and closure proof.
- What it does:
  Connects detection to remediation, not just alerts.
- How it helps:
  Demonstrates measurable safety improvement and closes the loop.
- What we have that is similar:
  Audit logs and labeling; no action/task object lifecycle.
- Gap status:
  Completely missing.
- Necessity: 5/5
- Sellability: 5/5
- Build complexity: Medium
- Why customers care:
  Buyers pay for outcomes, not notifications.

### [ ] 4. Root Cause Analysis (RCA) Toolkit
- What it is:
  Built-in investigation templates (5 Whys, Fishbone) plus evidence linking.
- What it does:
  Structures investigations and recurrence prevention.
- How it helps:
  Improves quality of investigations and compliance reporting.
- What we have that is similar:
  Image evidence storage (crop/full-frame), manual notes possible, no structured RCA workflow.
- Gap status:
  Mostly missing.
- Necessity: 4/5
- Sellability: 4/5
- Build complexity: Medium
- Why customers care:
  Critical for regulated and enterprise environments.

### [ ] 5. Integrations Layer (Teams/Slack + BI + EHS)
- What it is:
  Productized connectors and webhooks into collaboration, BI, and EHS systems.
- What it does:
  Pushes incidents, tasks, and KPI data to existing enterprise workflows.
- How it helps:
  Speeds deployment and reduces change management friction.
- What we have that is similar:
  REST API exists; no packaged connectors/playbooks.
- Gap status:
  Partially present.
- Necessity: 5/5
- Sellability: 5/5
- Build complexity: Medium
- Why customers care:
  Integration readiness strongly influences purchase decisions.

## P1 - Near-Term Expansion (Strong Differentiators)

### [ ] 6. Digital Permit-to-Work (ePTW)
- What it is:
  Digital permit workflows (hot work, confined space, electrical, excavation, etc.) with approvals/signatures.
- What it does:
  Controls high-risk work before execution.
- How it helps:
  Reduces paperwork delays and strengthens compliance.
- What we have that is similar:
  None in current product.
- Gap status:
  Completely missing.
- Necessity: 4/5
- Sellability: 5/5
- Build complexity: High
- Why customers care:
  Very strong in construction, energy, and heavy industry tenders.

### [ ] 7. Inspection and Audit Workflows
- What it is:
  Checklist templates, scheduled inspections, follow-up tracking, and audit closure.
- What it does:
  Brings preventive routines into the same platform as detections.
- How it helps:
  Improves leading indicators and compliance posture.
- What we have that is similar:
  Basic logs and reports; no checklist/scheduling system.
- Gap status:
  Completely missing.
- Necessity: 4/5
- Sellability: 4/5
- Build complexity: Medium
- Why customers care:
  Common requirement for replacing fragmented tools.

### [ ] 8. Expanded Detection Catalog (Beyond Current 5 Types)
- What it is:
  Add scenario modules such as area control, restricted access, housekeeping, vehicle behavior, ergonomics.
- What it does:
  Increases use cases per site and reduces single-feature dependence.
- How it helps:
  Raises contract value and cross-sell potential.
- What we have that is similar:
  Existing models: PPE, smoking, phone, fire/smoke, fall.
- Gap status:
  Partially present.
- Necessity: 4/5
- Sellability: 5/5
- Build complexity: High
- Why customers care:
  Buyers compare module breadth directly against competitors.

### [ ] 9. Virtual Geofence and Dynamic Zone Rules
- What it is:
  Rule engine for restricted zones tied to context (machine state, movement, shift).
- What it does:
  Triggers warnings/interventions based on spatial risk.
- How it helps:
  Prevents high-severity incidents in machine/vehicle environments.
- What we have that is similar:
  Temporal gating and model detections; no dynamic geofence control.
- Gap status:
  Mostly missing.
- Necessity: 4/5
- Sellability: 5/5
- Build complexity: High
- Why customers care:
  High perceived innovation and strong safety ROI story.

### [ ] 10. Multi-Site Governance Dashboard
- What it is:
  Portfolio-level view across sites, shifts, and zones.
- What it does:
  Enables benchmarking and centralized governance.
- How it helps:
  Supports enterprise rollout and leadership reporting.
- What we have that is similar:
  Single deployment dashboard and metrics endpoint.
- Gap status:
  Mostly missing.
- Necessity: 4/5
- Sellability: 4/5
- Build complexity: Medium
- Why customers care:
  Required by regional/global EHS leaders.

## P2 - Advanced/Strategic Capabilities

### [ ] 11. PLC / Hard-Stop Intervention
- What it is:
  Safety integration that can stop equipment automatically when risk is detected.
- What it does:
  Moves from alerting to real-time intervention.
- How it helps:
  Can prevent severe injuries when reaction time is critical.
- What we have that is similar:
  Alerting pipeline exists; no machine-control integration.
- Gap status:
  Completely missing.
- Necessity: 3/5
- Sellability: 5/5
- Build complexity: High
- Why customers care:
  Major differentiator for high-risk operations.

### [ ] 12. Mobile-First Field Operations App
- What it is:
  Mobile workflows for triage, actions, inspections, and approvals (with offline support).
- What it does:
  Lets supervisors operate from shop floor/field.
- How it helps:
  Faster response and higher workflow adoption.
- What we have that is similar:
  Browser UI; no dedicated mobile operations layer.
- Gap status:
  Mostly missing.
- Necessity: 3/5
- Sellability: 4/5
- Build complexity: High
- Why customers care:
  Essential for distributed field teams.

### [ ] 13. Privacy-by-Design Controls (De-identification)
- What it is:
  Optional face/body anonymization, strict retention controls, and privacy policy controls by site.
- What it does:
  Reduces legal/privacy friction in deployments.
- How it helps:
  Faster approvals with legal/compliance stakeholders.
- What we have that is similar:
  Evidence storage and logs, but no explicit de-identification feature set.
- Gap status:
  Mostly missing.
- Necessity: 4/5
- Sellability: 4/5
- Build complexity: High
- Why customers care:
  Increasingly required in privacy-sensitive regions.

### [ ] 14. AI Copilot for Safety Analytics
- What it is:
  Natural language assistant for querying incidents, trends, causes, and report drafting.
- What it does:
  Makes analytics self-serve for non-technical users.
- How it helps:
  Speeds decisions and report generation.
- What we have that is similar:
  Precision metric endpoint and API data; no NL analytics assistant.
- Gap status:
  Completely missing.
- Necessity: 3/5
- Sellability: 4/5
- Build complexity: Medium
- Why customers care:
  High demo appeal and executive visibility.

### [ ] 15. Leading Indicator Safety Scoring
- What it is:
  Composite risk score from behavior trends, repeat events, and response timeliness.
- What it does:
  Shifts teams from reactive incidents to proactive prevention.
- How it helps:
  Better management KPI alignment.
- What we have that is similar:
  Incident precision metric by type.
- Gap status:
  Partially present.
- Necessity: 4/5
- Sellability: 4/5
- Build complexity: Medium
- Why customers care:
  Easy way for leadership to track safety maturity.

## Suggested Build Sequence

### Phase 1 (0-6 weeks)
- [ ] Incident lifecycle engine
- [ ] Corrective actions module
- [ ] SLA timers and escalation
- [ ] Teams/Slack webhook integration

### Phase 2 (6-12 weeks)
- [ ] RCA toolkit
- [ ] Inspection/audit workflows
- [ ] Multi-site dashboard
- [ ] BI connector starter pack (Power BI/Tableau export)

### Phase 3 (12+ weeks)
- [ ] ePTW
- [ ] Virtual geofence
- [ ] PLC hard-stop pilot
- [ ] Privacy controls and de-identification

## Product Positioning Notes (Sales Narrative)
- Immediate story:
  "From detection to resolution" (lifecycle + actions + SLA) is the strongest sellable package.
- Enterprise story:
  "Fits your stack" (integrations + multi-site reporting) removes common buying blockers.
- Strategic story:
  "Prevention, not just alerts" (geofence, hard-stop, leading indicators) differentiates against basic AI camera tools.

## Definition of Done Template (Use per TODO)
- [ ] Problem statement approved
- [ ] UX flow finalized
- [ ] API contract finalized
- [ ] Data model migration complete
- [ ] Role/permission checks added
- [ ] Audit logging complete
- [ ] KPI impact measured (baseline vs post-release)
- [ ] Sales enablement material updated
