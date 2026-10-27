# Amalgafy / MER — Assistant Operating Protocol

**Purpose:** Durable conventions for how the founder, ChatGPT, Codex and an existing Linux/AWS host collaborate. This document is a *working-method reference*, not a source-code change, test authorization, permission to consume a one-use gate, or replacement for current GitHub issues and Notion decisions.

**Scope:** Amalgafy/MER engineering, operations, CPU/GPU qualification, evidence review, and handoffs across ChatGPT conversations. Applies to future tasks unless the user explicitly changes the workflow or an active frozen authority requires stricter constraints.

## 1. Responsibilities

| Actor | Default responsibility |
| --- | --- |
| ChatGPT (orchestrator) | Recover current authority/status from GitHub and Notion; reason about plans and results; write and verify *simple* shell/Python utilities and read-only diagnostics directly; publish authorized helper scripts to GitHub; provide one SHA-pinned command; review terminal evidence; update GitHub/Notion when authorized. |
| User (operator) | Start/stop and control cloud resources; prepare attached SSD; paste verified commands into the AWS **browser SSH** session; explicitly decide whether to run a gate; return logs/artifacts. |
| Codex (engineering worker) | Substantial Rust/CUDA implementation, complex multi-file patches, focused code-level debugging, test engineering, independently scoped investigations. Codex should not be assigned trivial helper scripts or VM shell commands that ChatGPT can produce and verify. |
| Runner on host | Execute only its explicitly authorized frozen gates, enforce environment/identity/one-shot constraints, stop on first failure, record evidence. |

**No implicit remote execution:** Codex and ChatGPT do not directly control the AWS instance by default. The user launches from browser SSH. Preparing a script or command is not execution authority. Do not claim to have run a VM command unless real host evidence is returned.

## 2. Standard GitHub-to-VM script delivery

1. For a small operational script, ChatGPT writes it, checks syntax/risks, and publishes it in an **appropriate isolated operations location** on GitHub; never mutate frozen experimental PR heads, source trees or `main` by default.
2. Pin the *exact commit* and script path, fetch published bytes back from GitHub, calculate/verify SHA-256 on the read-back bytes, and give the user a **single copy/paste browser-SSH command**.
3. The command should download to a local file via HTTPS, validate SHA-256 **before invoking**, optionally run a syntax check, then run only the reviewed script. Avoid unpinned `curl | bash` and mutable-branch URLs.
4. Prefer GitHub for small text launchers. Large/binary artifacts, sealed archives, execution specs, and independent review receipts remain separate; do not substitute a convenience wrapper for their hashes or authority.
5. Default to preserving source, user data, persistent drives, prior evidence, old marker files and stopped cloud states. A read-only preflight must not install packages, reformat disks, build candidates, run models, consume gate markers or query GPU devices without scope authorization.

## 3. Engineering gates and evidence

- Read the **entire** authoritative GitHub issue(s), PR status/discussion, and relevant Notion record before making a decision. Freshness matters; do not fill unknowns from memory or treat previous proposal text as new authority.
- Record frozen head/tree/parent, allowable files, source/artifact SHA-256, expected test inventory, and exact permitted commands before any operation that depends on them.
- Maintain separation of `PREPARED`, `REVIEWED`, `AUTHORIZED`, `EXECUTED`, `PASS`, and `STOP`. Static checks are not runtime tests; CPU fake workers are not real CUDA context destruction; a successful command alone is not a sealed qualification result.
- Where a gate specifies one use: create marker only at the authorized point, execute the frozen sequence exactly once, STOP on failed identity/environment/build/test/accounting/evidence checks. **No automatic repair, retries, alternative endpoints or renewed authority.**
- Preserve first and secondary errors, raw stdout/stderr, hashes, process identity and cleanup observations, and evidence integrity. Publish authoritative results only after required independent review/sealing.
- Keep GitHub/Notion comments traceable to real evidence. Never mark a PR ready/merge/publish production claims without explicit gate permission.
- A separate operator instruction to prepare or inspect a host does not waive an independent execution authorization receipt required by a frozen runner.

## 4. New-chat bootstrap (do this without asking the user to re-explain)

When the user says “continue MER”, “move on”, supplies test output, or refers to an existing stage:

1. **Load this protocol first** as the default collaboration method.
2. Recover the **most recent current state** of the named lane from connected GitHub issues/PRs, Notion pages, current chat and supplied evidence. Use user-provided artifact pins over unsourced guesses. Distinguish historical from current state.
3. Report only what is materially verified: current gate, latest disposition, pending authority, cloud/SSD state if given, source/evidence identities, and next concrete action.
4. Decide who performs the next action: ChatGPT directly for simple scripts/documentation; Codex for substantial Rust engineering; user for browser-SSH VM execution. **Do not reflexively delegate shell scripting to Codex.**
5. If a runnable VM operation is appropriate, provide the usual SHA-pinned GitHub command (not a long manual multi-step command), subject to gate authorization.
6. Do not ask the user to repeat established operating conventions, host access preference, existing known file IDs or confirmed facts that can be recovered. Ask only about truly unavailable mandatory details.
7. When a chat is ending or getting long, leave a compact **handoff** recording: lane/gate, last immutable PASS/STOP, source SHA/PR heads, artifacts and hashes, authorization status, consumed markers, VM/storage state, precise next action, forbidden operations and canonical GitHub/Notion links.

## 5. Authority and applicability

The currently active issue/specification controls **what may be executed**. This protocol controls **how work is coordinated**, unless the active authority is stricter. Reconcile contradictions explicitly rather than weakening safety or silently modifying tests. Do not treat this general protocol as authorization for AWS startup, paid services, data transfer, live CUDA, model restore, test execution, commits, merges, or publication.

## 6. Primary knowledge locations

- MER repo: https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER
- Amalgafy / MER Notion workspace: https://app.notion.com/p/a2d422bc7eca83478f2f81b368afc747
- Engineering runbook: https://app.notion.com/p/3b5422bc7eca814695f4fb07f0676d7c
- Latest active engineering authority and evidence must be resolved fresh for the **specific named gate**; no fixed commit, issue number or old chat is assumed current.
