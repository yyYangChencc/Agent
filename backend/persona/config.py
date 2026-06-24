from __future__ import annotations
from dataclasses import dataclass, field
from persona.opinion.scale import JIANG_PING_TOPIC, OPINION_NEUTRAL


@dataclass
class AgentConfig:
    # LLM
    llm_model: str = "deepseek-v4-pro-1"
    embedding_model: str = "text-embedding-3-small"
    llm_timeout_seconds: float = 30.0
    llm_connect_timeout_seconds: float = 5.0
    llm_max_retries: int = 0
    embedding_timeout_seconds: float = 15.0
    embedding_connect_timeout_seconds: float = 5.0
    embedding_max_retries: int = 0

    # World
    observation_radius: int = 5
    simulation_step_limit: int = 100
    # Euclidean distance squared threshold for eating: dx^2 + dy^2 <= eat_distance_sq
    eat_distance_sq: float = 2.0

    # Agent needs
    max_history: int = 12
    memory_top_k: int = 5
    memory_focus_bonus_k: int = 2
    memory_max_top_k: int = 8
    satiety_threshold: float = 30.0
    relax_threshold: float = 30.0
    money_threshold: float = 15.0
    relax_increase_rate: float = 1.2
    relax_decay_rate: float = 0.8
    relax_moving_usage: float = 1.0
    satiety_decay_rate: float = 0.5
    sleep_time: int = 8
    sleep_relax_recover: float = 60.0
    urgency_floors: dict[str, float] = field(default_factory=lambda: {
        "satiety": 0.08,
        "relax": 0.08,
        "money": 0.05,
    })
    urgency_gap_weights: dict[str, float] = field(default_factory=lambda: {
        "satiety": 1.20,
        "relax": 1.10,
        "money": 1.00,
    })
    need_layers: dict[str, list[str]] = field(default_factory=lambda: {
        "physiological": ["satiety", "relax"],
        "safety": ["money"],
    })
    need_layer_order: list[str] = field(default_factory=lambda: [
        "physiological",
        "safety",
    ])

    # Need pressure layer: satisfaction -> gap -> pressure_memory -> load_saturation -> effective_pressure
    need_pressure_dt: float = 1.0
    pressure_default_recovery_rate: float = 0.15
    pressure_default_load_kappa: float = 4.0
    pressure_default_current_weight: float = 0.60
    pressure_default_residual_weight: float = 0.15
    pressure_default_amplification_weight: float = 0.25
    pressure_recovery_rates: dict[str, float] = field(default_factory=lambda: {
        "satiety": 0.20,
        "relax": 0.15,
        "money": 0.05,
    })
    pressure_load_kappas: dict[str, float] = field(default_factory=lambda: {
        "satiety": 4.0,
        "relax": 4.0,
        "money": 8.0,
    })
    pressure_current_weights: dict[str, float] = field(default_factory=lambda: {
        "satiety": 0.60,
        "relax": 0.60,
        "money": 0.60,
    })
    pressure_residual_weights: dict[str, float] = field(default_factory=lambda: {
        "satiety": 0.15,
        "relax": 0.15,
        "money": 0.15,
    })
    pressure_amplification_weights: dict[str, float] = field(default_factory=lambda: {
        "satiety": 0.25,
        "relax": 0.25,
        "money": 0.25,
    })

    # Psychological assessment stage
    psychological_assessment_interval: int = 10
    psychological_pressure_default_threshold: float = 0.5
    psychological_pressure_thresholds: dict[str, float] = field(default_factory=lambda: {
        "satiety": 0.5,
        "relax": 0.5,
        "money": 0.5,
    })
    psychological_recovery_pressure_threshold: float = 0.25
    psychological_mediator_decay_rate: float = 0.50
    psychological_mediator_clear_threshold: float = 0.05
    psychological_assessment_mode: str = "llm"
    psychological_llm_fallback_to_rule: bool = True

    # Social platform
    popularity_like_weight: float = 1.0
    popularity_repost_weight: float = 2.0
    popularity_comment_weight: float = 1.0
    default_online_trust: float = 0.5

    # Opinion assessment
    initial_opinion: float = OPINION_NEUTRAL
    default_opinion_topic: str = JIANG_PING_TOPIC
    opinion_assessment_history_limit: int = 200
    opinion_assessment_mode: str = "llm"
    opinion_assessment_recent_history: int = 8
    opinion_assessment_recent_social: int = 5

    # Micro-reflection
    micro_reflect_interval: int = 3
