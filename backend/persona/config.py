from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    # LLM
    llm_model: str = "gpt-5.4mini"
    embedding_model: str = "text-embedding-3-small"

    # World
    observation_radius: int = 5
    # Euclidean distance squared threshold for eating (dx²+dy² <= eat_distance_sq)
    eat_distance_sq: float = 2.0

    # Agent
    max_history: int = 12
    memory_top_k: int = 5
    # need 范围：satiety / relax 为 [0, 100]，money 无上限
    satiety_threshold: float = 30.0
    relax_threshold: float = 30.0
    money_threshold: float = 0.3
    relax_increase_rate: float = 10.0   # 任务为 "none"时每 tick relax 增加的量（满足 relax 需求的速率）
    relax_decay_rate: float = 2.0       # relax 需求的自然衰减速率（每 tick 减少的量）
    relax_moving_usage: float = 2.0     # agent 移动一步消耗的 relax 量
    satiety_decay_rate: float = 0.5     # 饱腹度的自然降低速率（每 tick 饱腹度降低的量）
    sleep_time: int = 8                 # 睡觉恢复的时间（单位：tick），睡觉时 agent 不会移动，休息时间结束后恢复 relax
    sleep_relax_recover: float = 50.0   # 每次睡觉结束后恢复的 relax 量

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
