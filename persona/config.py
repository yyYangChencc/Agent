from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    # LLM
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # World
    observation_radius: int = 5
    # Euclidean distance squared threshold for eating (dx²+dy² <= eat_distance_sq)
    eat_distance_sq: float = 2.0

    # Agent
    max_history: int = 20
    memory_top_k: int = 5
    hunger_threshold: float = 0.3
    relax_threshold: float = 0.3
    relax_decay_rate: float = 0.1   # relax demand drop per tick while task is "none"

    # Social platform
    popularity_like_weight: float = 1.0
    popularity_repost_weight: float = 2.0
    popularity_comment_weight: float = 1.0

    # Opinion propagation
    initial_opinion: float = 0.5        # starting opinion value, range [0, 1]
    online_opinion_lr: float = 0.1      # α: learning rate for online update
    self_confidence: float = 0.5        # σ: resistance to external influence (higher = more resistant)
    offline_opinion_lr: float = 0.1     # β: conformity strength for offline update
    offline_update_interval: int = 3    # m: offline update runs every m ticks
    default_online_trust: float = 0.5   # τ default for online trust between agents
    default_offline_trust: float = 0.5  # T default for offline trust between agents
    friend_trust_threshold: float = 0.6 # offline_trust >= this to count as an offline neighbor/friend

    # Micro-reflection
    micro_reflect_interval: int = 3    # trigger micro-reflection after N ticks with no demand progress
