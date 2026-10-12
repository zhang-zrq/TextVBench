# Security

Keep VLM credentials in environment variables or ignored local configuration
files. `.env.example` contains placeholders; the application does not load
`.env` automatically. Never commit actual credentials, model checkpoints,
generated videos, or evaluation/debug output.

Both review servers bind to `127.0.0.1` by default and have no authentication.
Use them for local review. Any remote deployment needs an authenticated proxy
and a review of the exposed content.

New debug configurations redact named credentials before writing them to disk.
Review responses also redact known credentials, paths, and error endpoints.
These filters do not make arbitrary model responses, images, or older debug
bundles safe to publish. Inspect and sanitize any artifact before sharing it.

If a credential is exposed, revoke or rotate it first. Removing a file from the
latest commit does not remove it from Git history.

Do not disclose active credentials or private data in a public issue. Use
GitHub's private vulnerability reporting feature when available; otherwise
open an issue requesting a private contact without including sensitive details.
