# Institutional Disaster Recovery Protocol

## Core Philosophy
The News-Driven Alpha platform must survive arbitrary state corruption, latency spikes, and silent failures without losing determinism. This protocol explicitly outlaws automated overwrites of corrupted ledgers, instead demanding immutable restoration from cryptographic snapshots.

## 1. Immutable Backup Snapshots
- `disaster_recovery_protocol.py` is executed to snapstate `outputs/` and `logs/` files.
- Each backup is appended with its MD5 hash (e.g., `live_tracking.csv_<hash>.bak`) and placed in `archive/disaster_recovery_backups/`.
- **Rule**: Backups are append-only. Historical backups are never deleted.

## 2. Restoration Governance
- **Corrupted Manifests**: If a manifest is corrupted, it MUST be manually quarantined. The script will explicitly refuse to overwrite an existing non-empty manifest to prevent data destruction.
- **Verification**: Following any restoration, `audit_scripts/operational_persistence_audit.py` MUST be run to re-certify UUID continuity and timestamp monotonicity.
- **Rollback Continuity**: Restoring an old manifest effectively reverts the system execution state. This must be matched by a model rollback (`rollback_recovery_audit.py`) if the model was promoted in the missing period.

## 3. SAFE MODE Restoration
If the entire environment is compromised:
1. Clear the corrupted execution artifacts.
2. Run `python disaster_recovery_protocol.py` (which includes `restore_manifest()` capability).
3. The next pipeline run will natively intercept the missing forward state and re-align via SAFE MODE if necessary, protecting capital until determinism is manually certified via `full_pipeline_regression.py`.
