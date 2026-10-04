# Security

- The repository does not include broker connectivity or live-capital execution.
- Local environment files such as `.env` are ignored by Git.
- Historical GDELT BigQuery reconstruction uses local Google Application Default Credentials / project configuration. Credential files, access tokens, and quota-project secrets are **not** stored in this repository.
- Generated historical datasets and model/result artifacts remain outside version control.
- No private API keys are required to inspect the committed methodology, documentation, frozen specifications, or tests.
- If additional private data-provider credentials are used in future work, they must remain outside version control.
