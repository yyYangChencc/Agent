from __future__ import annotations
from dataclasses import dataclass, field
from persona.opinion.scale import JIANG_PING_TOPIC, OPINION_NEUTRAL


@dataclass
class AgentConfig:
    # LLM
    llm_model: str = "gpt-5.5"
    embedding_model: str = "text-embedding-3-small"
    llm_timeout_seconds: float = 120.0
    llm_connect_timeout_seconds: float = 5.0
    llm_max_retries: int = 0
    llm_max_concurrent_requests: int = 5
    llm_rate_limit_retries: int = 3
    llm_rate_limit_backoff_seconds: float = 2.0
    embedding_timeout_seconds: float = 15.0
    embedding_connect_timeout_seconds: float = 5.0
    embedding_max_retries: int = 0
    embedding_max_concurrent_requests: int = 5
    embedding_rate_limit_retries: int = 2
    embedding_rate_limit_backoff_seconds: float = 1.0

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
    # 记忆检索超时后直接返回已有结果，避免 embedding/Chroma 慢调用拖住 tick。
    memory_retrieval_timeout_seconds: float = 8.0
    # 相同记忆查询在短时间内复用结果，减少相邻 tick 的重复 SQLite/Chroma 检索。
    memory_retrieval_ttl_ticks: int = 3
    # LLM planner 默认开启；规则短路命中时不调用 planner。
    memory_planner_enabled: bool = True
    memory_planner_skip_when_observation_sufficient: bool = True
    satiety_threshold: float = 30.0
    relax_threshold: float = 30.0
    money_threshold: float = 15.0
    belonging_threshold: float = 30.0
    esteem_threshold: float = 30.0
    self_actualization_threshold: float = 30.0
    relax_increase_rate: float = 1.2
    # 保留旧字段兼容历史配置；当前规则不再按 tick 自动衰减 relax。
    relax_decay_rate: float = 0.0
    relax_moving_usage: float = 1.0
    satiety_decay_rate: float = 0.5
    sleep_time: int = 8
    sleep_relax_recover: float = 60.0
    # 需求事件规则参数。LLM 只给出行为和元数据，满足度变化由这些规则落地。
    social_feedback_agreement_threshold: float = 0.3
    social_like_belonging_delta: float = 0.3
    social_like_esteem_delta: float = 0.6
    social_dislike_belonging_delta: float = -0.3
    social_dislike_esteem_delta: float = -0.6
    social_comment_positive_belonging_delta: float = 0.4
    social_comment_positive_esteem_delta: float = 0.8
    social_comment_negative_belonging_delta: float = -0.4
    social_comment_negative_esteem_delta: float = -0.8
    work_esteem_delta: float = 0.3
    belonging_passive_decay_window_ticks: int = 10
    belonging_passive_decay_delta: float = -0.5
    esteem_passive_decay_window_ticks: int = 20
    esteem_passive_decay_delta: float = -0.3
    self_actualization_first_building_delta: float = 1.0
    self_actualization_first_building_kind_delta: float = 1.5
    self_actualization_first_topic_post_delta: float = 0.5
    self_actualization_task_completion_delta: float = 1.0
    self_actualization_repetition_window_ticks: int = 15
    self_actualization_repetition_decay_delta: float = -0.5
    self_actualization_low_level_tools: list[str] = field(default_factory=lambda: [
        "move",
        "eat",
        "sleep",
        "enter_building",
        "exit_building",
        "company",
        "food_shop",
        "playground",
    ])
    urgency_floors: dict[str, float] = field(default_factory=lambda: {
        "satiety": 0.08,
        "relax": 0.08,
        "money": 0.05,
        "belonging": 0.04,
        "esteem": 0.04,
        "self_actualization": 0.03,
    })
    urgency_gap_weights: dict[str, float] = field(default_factory=lambda: {
        "satiety": 1.20,
        "relax": 1.10,
        "money": 1.00,
        "belonging": 0.90,
        "esteem": 0.85,
        "self_actualization": 0.80,
    })
    need_layers: dict[str, list[str]] = field(default_factory=lambda: {
        "physiological": ["satiety", "relax"],
        "safety": ["money"],
        "belonging": ["belonging"],
        "esteem": ["esteem"],
        "self_actualization": ["self_actualization"],
    })
    need_layer_order: list[str] = field(default_factory=lambda: [
        "physiological",
        "safety",
        "belonging",
        "esteem",
        "self_actualization",
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
        "belonging": 0.10,
        "esteem": 0.10,
        "self_actualization": 0.08,
    })
    pressure_load_kappas: dict[str, float] = field(default_factory=lambda: {
        "satiety": 4.0,
        "relax": 4.0,
        "money": 8.0,
        "belonging": 6.0,
        "esteem": 6.0,
        "self_actualization": 8.0,
    })
    pressure_current_weights: dict[str, float] = field(default_factory=lambda: {
        "satiety": 0.60,
        "relax": 0.60,
        "money": 0.60,
        "belonging": 0.55,
        "esteem": 0.55,
        "self_actualization": 0.50,
    })
    pressure_residual_weights: dict[str, float] = field(default_factory=lambda: {
        "satiety": 0.15,
        "relax": 0.15,
        "money": 0.15,
        "belonging": 0.15,
        "esteem": 0.15,
        "self_actualization": 0.15,
    })
    pressure_amplification_weights: dict[str, float] = field(default_factory=lambda: {
        "satiety": 0.25,
        "relax": 0.25,
        "money": 0.25,
        "belonging": 0.20,
        "esteem": 0.20,
        "self_actualization": 0.20,
    })

    # Psychological assessment stage
    psychological_assessment_interval: int = 10
    psychological_pressure_default_threshold: float = 0.5
    psychological_pressure_thresholds: dict[str, float] = field(default_factory=lambda: {
        "satiety": 0.5,
        "relax": 0.5,
        "money": 0.5,
        "belonging": 0.5,
        "esteem": 0.5,
        "self_actualization": 0.5,
    })
    psychological_recovery_pressure_threshold: float = 0.25
    psychological_mediator_decay_rate: float = 0.50
    psychological_mediator_clear_threshold: float = 0.05
    psychological_assessment_mode: str = "llm"
    psychological_llm_fallback_to_rule: bool = True
    # 消融实验使用；关闭后心理评测仍记录，但不进入行为/观念 prompt。
    dynamic_role_card_enabled: bool = True

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
    # 观念评测采用事件触发，并用间隔兜底，减少无证据 LLM 调用。
    opinion_assessment_interval: int = 5
    opinion_assessment_triggered_only: bool = True
    opinion_max_delta_per_assessment: float = 0.25
    opinion_assessment_recent_social: int = 5
    post_opinion_scoring_mode: str = "llm"
    post_opinion_llm_fallback_to_rule: bool = True

    # Micro-reflection
    micro_reflect_interval: int = 3
