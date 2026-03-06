# Epic 9: Kin Interface — Agent Configuration & Management

## Vision

Rebrand "team" to "kin" throughout the UI. The Kin Interface is where users define, configure, and manage their agents. Concierge + Orienter remain always-on bootstrap agents (not configurable). All other agents are defined in the Kin section with configurable behavior.

## Research Findings

### Current State
- **7 hard-coded roles** in `role.ex`: lead, researcher, coder, reviewer, tester, concierge, orienter
- **No DB schema** for agent configurations — roles are static code
- **No UI** for agent management
- **Bootstrap agents** (concierge + orienter) hard-coded in `session.ex:706-741`
- **Team templates** exist in `.loomkin.toml` but are read-only, static
- **Agent spawning** only happens via: bootstrap (session start) or `team_spawn` tool (agent-initiated)

### Why Only Concierge + Orienter Show Up
The spawning flow research confirmed the pipeline is correct — but **no mechanism exists for spawning additional agents beyond bootstrap**. The concierge must use the `team_spawn` tool to create sub-teams with more agents. If the concierge doesn't decide to do that (or the LLM doesn't invoke the tool), no other agents appear.

This is the core UX gap: users can't pre-define which agents they want, and there's no "potency" signal to encourage spawning.

## Proposed Architecture

### 9.1: Kin Schema & Persistence

New `kin_agents` table:
```elixir
schema "kin_agents" do
  field :name, :string           # e.g. "code-reviewer"
  field :display_name, :string   # e.g. "Code Reviewer"
  field :role, Ecto.Enum, values: [:lead, :researcher, :coder, :reviewer, :tester]
  field :auto_spawn, :boolean, default: false   # spawn on session start
  field :potency, :integer, default: 50         # 0-100, how strongly to encourage spawning
  field :model_override, :string                # nil = use session default
  field :system_prompt_extra, :string           # appended to role's base prompt
  field :tool_overrides, :map                   # add/remove tools from role defaults
  field :budget_limit, :integer                 # per-session token budget (nil = unlimited)
  field :tags, {:array, :string}                # for grouping/filtering
  field :enabled, :boolean, default: true       # soft disable without deleting
  belongs_to :user, Loomkin.Schemas.User
  timestamps()
end
```

### 9.2: Potency System

Potency (0-100) controls how aggressively the system encourages spawning:
- **0-20 (dormant)**: Only spawned if explicitly requested by user or another agent
- **21-50 (available)**: Mentioned in concierge's context as available specialists
- **51-80 (eager)**: Injected into concierge's system prompt as recommended team members
- **81-100 (proactive)**: Auto-suggested to user after orientation completes ("I'd recommend spawning X for this project")

Implementation: On session start, after concierge + orienter spawn, inject a system message listing available kin agents with their potency levels. The concierge's prompt already handles team composition decisions.

### 9.3: Auto-Spawn Agents

Agents with `auto_spawn: true` are spawned alongside concierge + orienter during `maybe_spawn_bootstrap_agents/1`. These bypass potency — they always start.

### 9.4: Kin UI — Agent Builder/Editor

A new LiveView section (tab or modal) for managing agent definitions:
- List of defined agents with status indicators
- Create/edit form: name, role, potency slider, model picker, prompt customization
- Enable/disable toggle
- Auto-spawn checkbox
- Tag management for grouping
- Preview of effective configuration (base role + overrides)

### 9.5: Kin Dashboard (replaces "Team" section)

Rebrand team_dashboard_component.ex to kin_dashboard_component.ex:
- Show all defined agents (not just active ones)
- Active agents show live status from agent cards
- Dormant agents show as available with "spawn" button
- Budget overview per-agent and aggregate
- Health metrics

### 9.6: Agent Card Enhancements

Research questions for what to display on agent cards:
- **Current**: name, role, status (idle/working), current task, latest content, last tool, pending question
- **Candidates to add**:
  - Token usage / budget remaining (already tracked in CostTracker)
  - Message queue depth (priority + normal)
  - Iteration count in current loop
  - Model being used (especially after escalation)
  - Time in current state
  - Tools available to this agent
  - Context window usage percentage
  - Capability scores (from capabilities.ex ETS)
  - Sub-team membership indicator

### 9.7: Kin Naming Throughout UI

Rename all user-facing "team" references to "kin":
- Team Dashboard -> Kin Dashboard
- Team Activity -> Kin Activity
- team_spawn tool display -> "Spawning kin"
- PubSub topic display names
- Command palette entries

## Data Available for Agent Cards (from diagnostic logging)

The diagnostic `[Kin:]` logs now trace the complete agent visibility pipeline:

```
[Kin:agent]      → Agent GenServer init (name, role, team_id)
[Kin:spawn]      → Manager.spawn_agent result (success/failure with pid)
[Kin:roster]     → list_agents filter (what gets dropped and why)
[Kin:team_spawn] → team_spawn tool invocation (team name, roles, parent)
[Kin:UI]         → WorkspaceLive events (:team_available, :child_team_created,
                   refresh_roster, roster_agents, sync_cards_with_roster)
```

At each stage, the log includes team_id, agent names, and counts — making it possible to pinpoint exactly where agents get lost in the pipeline.

## Implementation Priority

1. **P0**: Diagnostic logging (DONE) — understand current agent visibility
2. **P1**: Agent card enhancements — show more useful data on existing cards
3. **P2**: Kin schema + auto-spawn — let users define agents that start automatically
4. **P3**: Potency system — influence which agents get spawned
5. **P4**: Kin UI — full agent builder/editor
6. **P5**: Rebrand team -> kin throughout UI
