# Incident Operations System Proposal

## 1. Incident Lifecycle Engine

Define a strict operational lifecycle:

- new -> triage -> assigned -> investigating -> confirmed/false_positive -> closed

Add:

- SLA timers by severity (for example high severity must be triaged in 5 minutes)
- Auto-escalation when SLA is breached (supervisor, then manager)
- Immutable transition history for compliance and audits

## 2. Labeling and Review Workflow

Use two labeling layers:

- Detection label (model output)
- Human review label (final truth)

Reviewer actions:

- Confirm violation
- Mark false positive
- Reclassify type and severity
- Add root-cause code
- Add corrective action code

Queue behavior:

- Group low-confidence incidents for batch review
- Pin high-severity incidents at the top

## 3. Duplicate and Same-Event Control

Reduce repeat spam by grouping events in a short time window.

Suggested grouping key:

- camera + event type + overlapping bbox/zone + time window

Output model:

- One master incident with child sightings

Benefits:

- Cleaner dashboard
- Cleaner paperwork
- Better KPI quality

## 4. Case File Model (Paperwork Backbone)

For every incident keep:

- Core facts: time, camera, zone, event type, severity
- Evidence pack: snapshots, full frame, short clip, annotations
- People fields: assigned owner, approver, witnesses
- Compliance fields: policy clause, risk score, required action, due date
- Closure pack: corrective action, verification note, attachments, sign-off

## 5. Automated Reports and Paperwork

Auto-generate:

- Shift summary
- Daily safety report
- Weekly KPI report
- CAPA form

Export formats:

- PDF
- CSV
- JSON

Scheduling:

- Role-based email digests
- Daily review digest
- Auto-filled report templates with minimal manual editing

## 6. Operations Dashboard Additions

Queues:

- Needs Triage
- Overdue
- Awaiting Approval

KPI cards:

- MTTA (mean time to acknowledge)
- MTTR (mean time to resolve)
- False positive rate by model and camera
- Repeat incident rate by zone

Reviewer metrics:

- Reviewed items per hour
- Backlog trend

## 7. Permissions and Audit

RBAC roles:

- Operator
- Supervisor
- EHS manager
- Auditor

Permission controls:

- Who can relabel
- Who can close incidents
- Who can edit historical records

Audit requirements:

- Full log for every state change and every field edit

## 8. Suggested API Surface

- POST /incidents/{id}/assign
- POST /incidents/{id}/label
- POST /incidents/{id}/close
- POST /incidents/{id}/merge
- GET /incidents/queue?status=triage&overdue=true
- POST /reports/generate
- GET /reports/{id}/download
- GET /kpi/ops

## 9. Phased Build Plan

1. Lifecycle state machine + assignment + audit trail
2. Duplicate grouping + review queue
3. Report templates + scheduled exports
4. SLA escalation + approval/sign-off workflow

## 10. Optional Next Step (Same-Person Suppression)

Lightweight approach without adding ReID model:

- Track detections per camera with IoU matching across recent frames
- Maintain track cache with track_id, last_seen, last_alert_ts
- Suppress repeat alerts while now - last_alert_ts is below cooldown

Later upgrade:

- Add ReID embeddings for stronger identity consistency across occlusion and movement
