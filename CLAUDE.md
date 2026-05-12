# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env  # then fill in OPENAI_API_KEY and OPENAI_BASE_URL

# Run the multi-agent simulation (5 agents, 10 ticks, opinion tracking)
python main.py

# Run the social subsystem demo
python test.py
```

All scripts must be run from the repository root — there are no `__init__.py` files, so relative imports depend on the working directory.

## Environment Variables

```
OPENAI_API_KEY=<key>
OPENAI_BASE_URL=<OpenAI-compatible endpoint, e.g. https://api.openai.com/v1>
```

The project uses `python-dotenv`. `main.py` uses `os.environ.get()` (safe on missing key); `test.py` uses `os.environ[]` (raises `KeyError` if missing).

## Architecture

This is a **LLM-driven multi-agent grid simulation** with a separate social media subsystem and a dual-layer opinion propagation model. Code lives under: `persona/` (core simulation), `social_sys/` (social platform), `world/` (environment), `tools/` (action operators).

### Starting a simulation

Use `SimulationRuntime` (`persona/runtime.py`) as the single entry point:

```python
from persona.runtime import SimulationRuntime
from world.objects import food

rt = SimulationRuntime.build()          # reads env vars, creates all services
rt.mem.reset_all()                       # wipe ChromaDB for a fresh run
a = rt.create_agent("agent_1", [3, 3], role="厨师", speaking_style="热情")
food("food_1", 1, 2, [10, 10], rt.world)
for _ in range(10):
    rt.world.step()
```

Pass a configured `AgentConfig` to `SimulationRuntime.build(config=...)` to override defaults. All tunable parameters live in `persona/config.py`.

`create_agent(agent_id, position, role="", speaking_style="")` — `role` and `speaking_style` are injected into every LLM system prompt. Each agent also carries `emotion: str` (default `"平静"`).

### Execution flow for each tick (`World.step()`)

```
World.step()
  └── ThreadPoolExecutor: _agent_full_step(agent) per agent
        ├── observer.observe(agent, radius)  → observation string
        ├── agent.step(obs)
        │     ├── recall(obs)            → ChromaDB keyword search → top-k memories
        │     ├── policy.decide(...)     → LLMPolicy builds prompt, calls OpenAI
        │     └── parser.parse_action()  → extracts <Action>JSON</Action>
        ├── world.execute(agent, action) → dispatches to Tool.run(),
        │                                  broadcasts Event to nearby agents
        ├── agent.append_trajectory()
        └── agent.get_reflect()          → Reflect.step():
                                              progress check → micro-reflect if stuck
                                              task completion check
                                              trajectory flush to memory if done
  └── opinion_updater.online_update()    → opinion shift from browsed posts
  └── opinion_updater.offline_update()  → (every m ticks) conformity with friends
  └── optional: _conversation_phase()   → multi-turn agent.conversation_step()
```

### Module responsibilities

| Module | Responsibility |
|--------|---------------|
| `persona/runtime.py` | Composition root — `build()` wires every service; `create_agent()` registers agents; `reset()` rebuilds world/platform |
| `persona/config.py` | `AgentConfig` dataclass — single source of truth for all parameters |
| `world/world.py` | Tick loop, parallel execution, action dispatch, event broadcast, opinion update hooks, conversation phase |
| `persona/agents/agent.py` | Agent state: position, demand, task, `current_focus`, `stuck_ticks`, history, opinion, trust dicts; drives `step()`, `social_step()`, `conversation_step()` |
| `persona/agents/policy.py` | `LLMPolicy`: builds prompt, calls LLM, returns parsed action string |
| `persona/agents/prompt.py` | 4 prompt builders (`WorldPromptBuilder`, `SocialPromptBuilder`, `ConversationPromptBuilder`, `ReflectPromptBuilder`). All prompts are Chinese, use `<Think>` + `<Action>JSON</Action>` format |
| `persona/agents/parser.py` | `ActionParser`: extracts `<Action>` block, validates JSON; returns `""` on failure |
| `persona/agent_memory/mem.py` | `MultiAgentMemoryManager`: one ChromaDB collection per agent; `smart_retrieve()` asks LLM for keywords then vector-searches |
| `persona/reflect/reflect.py` | Task lifecycle: progress tracking, micro-reflection trigger, task completion, trajectory flush |
| `persona/opinion/updater.py` | `OpinionUpdater`: online update (per tick, post-browsing) and offline update (every m ticks, friend conformity) |
| `persona/opinion/scorer.py` | `evaluate_opinion(content) -> float` — stub for LLM-based stance extraction (returns 0.5) |
| `tools/operator_tools.py` | `Operator` (move/eat/speak) and `SocialOperator` (send_post/comment/like/dislike) |
| `world/observer.py` | Scans grid for nearby entities, reads `agent.observed_events`, returns text |
| `social_sys/platform/platform.py` | Social feed: posts, follower-based distribution, social action dispatch |

### Action protocol

Agents communicate with the world via a text protocol:
- LLM must output `<Think>...</Think><Action>{"tool": "...", "args": {...}}</Action>`
- Empty `{}` = no-op; `ActionParser` returns `""` on any parse failure
- Tool failures are caught per-call and logged; they do not abort the tick

### Task and micro-reflection system

Tasks are managed in `persona/reflect/reflect.py`:
- **`_TASK_DEMAND_MAP`**: maps task name → demand key (e.g. `"eat something"` → `"hunger"`)
- **`_TASK_INITIAL_FOCUS`**: maps task name → default `current_focus` string injected when a task is assigned
- **Adding a new task**: add entries to both dicts; `_TASK_EXTRA_DESC` is optional

Each tick, `Reflect.step()` checks whether the task's demand is decreasing. If not, `agent.stuck_ticks` increments. When `stuck_ticks >= config.micro_reflect_interval` (default 3), a lightweight LLM call generates `<Insight>` (stored as memory) and `<Focus>` (updates `agent.current_focus`). The focus string appears in the next tick's action prompt.

`agent.update_demand(demand_key: str, demand_delta: float)` is the generic demand modifier — do not use keyword args.

### Opinion propagation system

Configured via `AgentConfig` fields (`initial_opinion`, `online_opinion_lr`, `self_confidence`, `offline_opinion_lr`, `offline_update_interval`, `default_online_trust`, `default_offline_trust`, `friend_trust_threshold`).

- **Online**: after each tick, `OpinionUpdater.online_update()` shifts opinion toward the trust-weighted average of seen posts: `Δ = α × (social_avg − opinion) × (1 − σ)`
- **Offline**: every m ticks, `offline_update()` applies conformity with agents whose `offline_trust >= friend_trust_threshold`; also applies conformity to demand values
- `like_post` / `dislike_post` adjust `online_trust` ±0.05
- `send_post` records `evaluate_opinion(content)` as the post's `opinion_index`

### Memory system

- Backend: `chromadb.PersistentClient` at `chroma_agents/`
- Per-agent collection: `agent_<id>_memory`
- `smart_retrieve()`: LLM extracts 3-5 keywords → vector search → top `memory_top_k` results
- `mem.reset_all()` wipes the DB; called at the start of `main.py`

## Known Issues (fix before extending)

- **`social_sys/agent/social_agent.py`**: calls `self.platform.excute(...)` — typo, method is `execute`. File is broken.
- **`persona/agent_memory/mem_operate_tools.py`**: references `agent.mem.agent_profile` which does not exist. Dead code.
- **`social_sys/post/post.py`**: imports `numpy` (unused) — not in `requirements.txt`.
- **`persona/opinion/scorer.py`**: `evaluate_opinion()` always returns 0.5; LLM-based implementation is a TODO.
- **Observer boundary**: `range(x, x+r)` is exclusive — the outermost ring of the observation radius is never scanned.
- **`logs/` not in `.gitignore`**: runtime log files can be accidentally committed.
