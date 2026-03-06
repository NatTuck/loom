# Epic 9: Permission Dashboard & Scalable Approval System

**Status**: PLANNED
**Priority**: CRITICAL — blocks multi-agent usability
**Created**: 2026-03-05
**Context**: Community feedback on permission UX when running 10-20 agents simultaneously

## Problem Statement

Loomkin's permission backend is architecturally sound (4-tier classification, non-blocking agent loop, session-scoped grants, path boundary enforcement), but the **UI is a single-slot modal** that silently drops concurrent requests. At scale (10+ agents), users either get blasted with individual permission prompts or — worse — requests are lost entirely because the `permission_request` assign overwrites previous values.

Community feedback highlights three core needs:
1. A **dashboard for intended actions** with approve/reject and comments
2. **Dynamic trust levels** users can tweak on the fly from the LiveView
3. **Natural checkpoints** from team coordination that reduce approval fatigue

## Current Architecture

### What Works
- `Permissions.Manager` — clean 4-tier classification (`:read`, `:write`, `:execute`, `:coordination`)
- Auto-approval for read + coordination tools (configurable in `.loomkin.toml`)
- Non-blocking agent loop — returns `{:pending, ...}`, agent pauses, resumes async on decision
- Session-scoped grants in DB with wildcard, exact-path, and directory-prefix matching
- Shell blocklist (13 dangerous patterns) + `safe_path!/2` project boundary enforcement
- Signal-based flow via Jido Signal Bus

### What's Broken
- **Single `permission_request` assign** in WorkspaceLive — concurrent requests overwrite each other
- **No queue** — lost requests mean agents hang indefinitely (60s timeout)
- **No comments** on approval decisions, no audit trail
- **No per-agent or per-role policies** — all agents treated equally
- **No runtime autonomy controls** — config changes require editing Elixir files
- **No permission batching** — each tool call = independent request
- **No revocation** — grants can't be revoked during a session
- **No pre/post-tool hooks** — no validation pipeline (tests, credo) before allowing execution

## Sub-Tasks

### 9.1: Permission Request Queue (P0 — CRITICAL)

**Goal**: Replace single-slot `permission_request` assign with a queue that never drops requests.

**Changes**:
- `workspace_live.ex`: Replace `permission_request: nil` with `pending_permissions: []` (list of pending requests)
- Each request gets a unique ID (already exists in PermissionRegistry)
- New requests append to the list, never overwrite
- Track request metadata: `%{id, tool_name, tool_path, source, agent_name, team_id, requested_at}`
- Handle race condition: if agent times out or cancels, remove from queue
- PubSub subscription: `"permissions:#{team_id}"` topic for live updates

**Acceptance Criteria**:
- 10 simultaneous permission requests all appear in state
- No request is silently dropped
- Timed-out requests auto-removed from queue
- Agent cancellation removes request from queue

**Estimated Effort**: Small

---

### 9.2: Approval Dashboard Component (P0 — CRITICAL)

**Goal**: Replace the single modal with a persistent approval panel showing all pending requests.

**Changes**:
- New `permission_dashboard_component.ex` — replaces `permission_component.ex` modal
- Layout: Collapsible panel (bottom or right side) with badge count of pending requests
- Each request row shows:
  - Agent name + role badge
  - Tool name (color-coded by category: green=read, yellow=write, red=execute)
  - File path (truncated, expandable)
  - Timestamp (relative: "5s ago")
  - Action buttons: Deny / Allow Once / Allow Always
- Batch actions toolbar:
  - "Approve All Reads" — auto-approve all pending `:read` requests
  - "Approve All for Agent X" — approve all pending requests from one agent
  - "Deny All" — deny everything pending
- Sort: newest first, with `:execute` requests pinned to top
- Empty state: "No pending approvals" with green checkmark

**Acceptance Criteria**:
- All pending permissions visible simultaneously
- Batch approve/deny works across multiple requests
- Category color-coding matches permission tiers
- Panel collapses to badge-only when no requests pending
- Responsive for different screen sizes

**Estimated Effort**: Medium

---

### 9.3: Approval Comments & Audit Trail (P1)

**Goal**: Let users add rationale when approving/denying, and log all decisions for review.

**Changes**:
- Add optional text input on approve/deny in dashboard component
- New schema `permission_audit_log`:
  ```
  permission_audit_logs
    id: binary_id
    session_id: references(sessions)
    team_id: string
    agent_name: string
    tool_name: string
    tool_path: string
    action: string (allow_once | allow_always | deny)
    comment: text (nullable)
    decided_at: utc_datetime
  ```
- `Permissions.Manager.record_decision/6` — writes to audit log on every decision
- Dashboard shows recent decisions (last 20) in a collapsible "Recent Decisions" section
- Export: "Copy audit log" button for session review

**Acceptance Criteria**:
- Users can type a comment before clicking approve/deny
- All decisions logged to DB with timestamp, agent, tool, path, action, comment
- Recent decisions visible in dashboard
- Comment is optional (empty string if not provided)

**Estimated Effort**: Medium

---

### 9.4: Per-Agent Trust Policies (P1)

**Goal**: Let users set permission policies per-agent or per-role from the UI.

**Changes**:
- New `trust_policy_component.ex` — accessible from agent card or dashboard settings
- Policy model (ETS, session-scoped):
  ```elixir
  %TrustPolicy{
    agent_name: "coder-1",        # or "*" for all agents
    role: :coder,                  # or :any for all roles
    tool_category: :write,         # :read | :write | :execute | :coordination | :all
    action: :auto_approve,         # :auto_approve | :ask | :deny
    scope: "/lib/"                 # path prefix, or "*" for all paths
  }
  ```
- `Permissions.Manager.check/4` consults trust policies before falling through to default behavior
- Policy priority: explicit agent policy > role policy > wildcard policy > default config
- Preset profiles:
  - **Strict**: Ask for everything (write + execute + out-of-project reads)
  - **Balanced** (default): Auto-approve reads + coordination, ask for writes + executes
  - **Autonomous**: Auto-approve reads + writes, ask only for execute (shell/git)
  - **Full Trust**: Auto-approve everything (equivalent to `:auto` mode, with warning)
- UI: Dropdown per-agent in agent card, or global preset selector in dashboard header

**Acceptance Criteria**:
- Users can set trust level per-agent from agent card
- Global preset selector changes policy for all agents
- Custom policies override presets
- Policy changes take effect immediately (no restart)
- Policies are session-scoped (reset on new session)

**Estimated Effort**: Large

---

### 9.5: Autonomy Slider & Runtime Controls (P1)

**Goal**: Give users a simple slider to adjust permission strictness on the fly.

**Changes**:
- Slider component in dashboard header or workspace toolbar
- 4 positions mapping to preset profiles from 9.4:
  - 1 (Strict) — Ask for everything
  - 2 (Balanced) — Ask for writes + executes
  - 3 (Autonomous) — Ask for executes only
  - 4 (Full Trust) — Auto-approve all (with confirmation dialog + warning)
- Slider change broadcasts updated policy to all agents via PubSub
- Visual indicator: colored bar (green → yellow → orange → red) showing current trust level
- Per-agent override: small lock icon on agent card to exclude from global slider

**Acceptance Criteria**:
- Slider updates all agent policies in real-time
- Visual feedback shows current trust level
- Full Trust requires explicit confirmation ("Are you sure?")
- Per-agent overrides persist when slider changes
- Slider position persists within session

**Estimated Effort**: Medium

---

### 9.6: Permission Batching (P1)

**Goal**: Group tool calls from a single LLM response into one approval prompt.

**Changes**:
- `agent_loop.ex`: After LLM responds with N tool calls, collect all permission checks before executing any
- New function `batch_permission_check/3`:
  - Takes list of `{tool_name, tool_path}` tuples
  - Returns `{auto_approved, needs_approval}` split
  - Auto-approved tools execute immediately
  - Remaining tools bundled into single `PermissionBatchRequest` signal
- Dashboard shows batch as grouped card:
  - "Agent Coder-1 wants to: file_write on /lib/foo.ex, file_edit on /lib/bar.ex, shell: mix test"
  - "Approve All / Deny All / Review Individually"
- Agent receives batch response, executes approved tools, skips denied ones

**Acceptance Criteria**:
- Single LLM response with 5 tool calls = 1 approval prompt (not 5)
- User can approve/deny individual tools within a batch
- Auto-approved tools in the batch execute without waiting
- Partial approval works (approve 3 of 5, deny 2)

**Estimated Effort**: Large

---

### 9.7: Permission Request Deduplication (P1)

**Goal**: When multiple agents request the same tool+path, deduplicate into one approval.

**Changes**:
- `workspace_live.ex`: Before appending to `pending_permissions`, check for existing request with same `{tool_name, tool_path}`
- If duplicate found, merge:
  - Show "Requested by: Coder-1, Researcher-2" (multiple agents)
  - Single approve/deny applies to all requesting agents
- Dedup key: `{tool_name, normalized_path}` (resolve relative paths)
- Dashboard shows merged count badge: "3 agents want file_read on /lib/config.ex"

**Acceptance Criteria**:
- Identical requests from 3 agents show as 1 dashboard entry
- Approving the merged entry unblocks all 3 agents
- Denying affects all 3 agents
- New requests for same tool+path merge into existing entry

**Estimated Effort**: Small

---

### 9.8: Grant Revocation & Management (P2)

**Goal**: Let users revoke grants and manage active permissions during a session.

**Changes**:
- New "Active Grants" section in permission dashboard
- Shows all current grants: tool, scope, agent, granted_at
- "Revoke" button per grant — deletes from `permission_grants` table
- "Revoke All" button — clears all grants for session
- "Revoke All for Agent X" — clears grants for specific agent
- Agent receives revocation notification via PubSub
- Next tool call by that agent re-triggers permission check

**Acceptance Criteria**:
- All active grants visible in dashboard
- Individual and bulk revocation works
- Revoked grants immediately take effect (next tool call asks again)
- Revocation logged in audit trail (from 9.3)

**Estimated Effort**: Medium

---

### 9.9: Pre/Post-Tool Validation Hooks (P2)

**Goal**: Run validation checks (tests, credo, compilation) as a condition for tool approval.

**Changes**:
- New `Permissions.Hook` behaviour:
  ```elixir
  @callback pre_tool(tool_name :: String.t(), tool_args :: map()) ::
    :allow | :deny | {:ask, reason :: String.t()}

  @callback post_tool(tool_name :: String.t(), result :: term()) ::
    :ok | {:warn, message :: String.t()} | {:rollback, reason :: String.t()}
  ```
- Built-in hooks:
  - `CompilationHook` — after file_write/file_edit, run `mix compile --warnings-as-errors`
  - `TestHook` — after file_write/file_edit, run `mix test --failed` (configurable)
  - `CredoHook` — after file_write/file_edit, run `mix credo --strict` on changed files
- Hook configuration in `.loomkin.toml`:
  ```toml
  [permissions.hooks]
  pre_tool = ["compilation"]
  post_tool = ["test", "credo"]
  block_on_failure = true  # deny tool if hook fails
  ```
- Dashboard shows hook results: green check / red X next to each approval
- If `block_on_failure = true`, failed hook auto-denies the tool with error message

**Acceptance Criteria**:
- Pre-tool hooks run before tool execution, can block
- Post-tool hooks run after execution, can warn or trigger rollback
- Hook results visible in dashboard
- Hooks configurable per-project in `.loomkin.toml`
- Hooks don't block read/coordination tools (only write/execute)

**Estimated Effort**: Large

---

### 9.10: Channel-Based Mobile Approvals (P2)

**Goal**: Push permission requests to Telegram/Discord for mobile approval (builds on Epic 7).

**Changes**:
- Extend `Channels.Bridge` to forward permission requests to bound channel
- Telegram: Inline keyboard with "Allow Once / Allow Always / Deny" buttons
- Discord: Button components with same options
- `PermissionRegistry` already supports channel-based approval (ETS store with request IDs)
- Wire up: permission signal → Bridge → channel adapter → inline buttons → callback → agent resume
- Rate limit: max 5 permission requests per minute per channel (prevent spam)
- Fallback: if channel user doesn't respond in 2 minutes, request stays in LiveView dashboard

**Acceptance Criteria**:
- Permission requests appear as interactive messages in Telegram/Discord
- Tapping a button resolves the permission (agent resumes)
- Rate limiting prevents channel spam
- Timeout falls back to LiveView dashboard
- Works alongside (not instead of) LiveView dashboard

**Estimated Effort**: Medium (depends on Epic 7 completion)

---

## Dependency Graph

```
9.1 (Queue) ─────────────────┐
                              ├──> 9.2 (Dashboard) ──> 9.3 (Comments/Audit)
                              │                    ──> 9.7 (Dedup)
                              │                    ──> 9.8 (Revocation)
                              │
9.4 (Trust Policies) ────────>├──> 9.5 (Autonomy Slider)
                              │
9.1 (Queue) ─────────────────>├──> 9.6 (Batching)
                              │
9.2 (Dashboard) + Epic 7 ───>└──> 9.10 (Mobile Approvals)

9.9 (Hooks) — independent, can be built in parallel
```

## Security Considerations

- **Default to `:session` mode** — `:auto` should require explicit opt-in with a warning
- **Enforce permission checks at `Tool.run/2`** — not just in agent loop (other call paths can bypass)
- **Strengthen shell blocklist** — handle whitespace/newline variations in fork bomb detection
- **Wildcard grants need guardrails** — `scope = "*"` on `:execute` tools should require extra confirmation
- **Out-of-project reads must be mandatory** — currently optional if `project_path` is nil
- **Grant expiration** — add optional TTL on "allow_always" grants (e.g., 1 hour)

## Community Feedback Addressed

| Feedback | Sub-Task |
|----------|----------|
| "Dashboard for intended actions with approve/reject" | 9.2 |
| "Approve or reject with a comment" | 9.3 |
| "Expose permissions in LiveView to tweak on the fly" | 9.4, 9.5 |
| "Blasted with permission requests with 10-20 agents" | 9.1, 9.6, 9.7 |
| "Hook system to gate dangerous operations" | 9.9 |
| "Mobile so I'm less tied to the desk" | 9.10 |
| "Trust mid-session vs requiring human checkpoints" | 9.4, 9.5 |
| "Natural pause points from team coordination" | 9.6 |
| "Let it run freely but block if compile/test degrades" | 9.9 |

## Files Involved

### Core (will be modified)
- `lib/loomkin_web/live/workspace_live.ex` — permission queue, routing, PubSub
- `lib/loomkin_web/live/permission_component.ex` — replaced by dashboard component
- `lib/loomkin/permissions/manager.ex` — trust policies, batching, hooks
- `lib/loomkin/teams/agent.ex` — batch permission responses, revocation handling
- `lib/loomkin/agent_loop.ex` — batch permission collection, hook integration
- `lib/loomkin/config.ex` — hook config, trust presets

### New Files
- `lib/loomkin_web/live/permission_dashboard_component.ex`
- `lib/loomkin_web/live/trust_policy_component.ex`
- `lib/loomkin/permissions/trust_policy.ex`
- `lib/loomkin/permissions/hook.ex`
- `lib/loomkin/permissions/hooks/compilation_hook.ex`
- `lib/loomkin/permissions/hooks/test_hook.ex`
- `lib/loomkin/permissions/hooks/credo_hook.ex`
- `lib/loomkin/schemas/permission_audit_log.ex`
- `priv/repo/migrations/*_create_permission_audit_logs.exs`

### Tests
- `test/loomkin/permissions/manager_test.exs`
- `test/loomkin/permissions/trust_policy_test.exs`
- `test/loomkin/permissions/hook_test.exs`
- `test/loomkin_web/live/permission_dashboard_test.exs`
