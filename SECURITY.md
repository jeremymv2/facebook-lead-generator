# Security Policy

## Reporting a leak

Do not open a public issue containing a credential, cookie, browser screenshot, private group URL,
or reproduction log with authentication data. Contact the repository owner privately and identify
the affected provider and file without pasting the secret again.

## If a credential is exposed

Treat any committed credential as compromised, including credentials pushed only to a private
repository.

1. Revoke or rotate the credential at its provider immediately.
2. Pause the affected integration until the replacement is installed safely.
3. Check branches, tags, pull requests, Actions logs and artifacts, forks, and other clones.
4. Remove the credential from the working tree and add a prevention rule if needed.
5. If removal from Git history is necessary, use a fresh clone and `git-filter-repo` only after
   rotation. Coordinate the rewrite with every clone owner to prevent recontamination.
6. Contact GitHub Support when cached views or pull-request references must be purged.

Deleting a file or rewriting Git history does not invalidate a copied credential. Rotation is the
first containment action.

## Prevention controls

- `.gitignore` excludes local settings and runtime artifacts.
- `pre-commit` rejects sensitive paths, private keys, large files, and Gitleaks findings.
- The pre-push hook runs Gitleaks against complete local history.
- GitHub Actions performs an independent full-history Gitleaks scan.
- Scanner output is redacted and report artifacts are not uploaded.
- Real Facebook credentials remain in the local persistent browser profile, outside this repository.

Repository administrators should also enable GitHub Secret Protection and push protection when the
account plan permits it, protect `main`, disallow direct and force pushes, and require the
`Gitleaks full history` status check before merge.
