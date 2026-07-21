from __future__ import annotations
from dataclasses import dataclass, field
from persona.opinion.scale import JIANG_PING_TOPIC, OPINION_NEUTRAL


@dataclass
class AgentConfig:
    # LLM
    llm_model: str = "deepseek-v4-flash"
    embedding_model: str = "text-embedding-3-small"
    llm_timeout_seconds: float = 120.0
    llm_connect_timeout_seconds: float = 5.0
    llm_max_retries: int = 0
    # 所有异步 LLM 请求共享的全局并发上限。
    llm_max_concurrent_requests: int = 20
    llm_rate_limit_retries: int = 3
    llm_rate_limit_backoff_seconds: float = 2.0
    llm_server_error_retries: int = 2
    llm_server_error_backoff_seconds: float = 1.0
    # 单次异步请求包含排队在内最多等待 270 秒。
    llm_total_timeout_seconds: float = 270.0
    # embedding 连接和读写均允许等待 60 秒；异步本地排队仍受 180 秒总预算约束。
    embedding_timeout_seconds: float = 60.0
    embedding_connect_timeout_seconds: float = 60.0
    embedding_max_retries: int = 0
    embedding_max_concurrent_requests: int = 20
    embedding_rate_limit_retries: int = 2
    embedding_rate_limit_backoff_seconds: float = 1.0
    embedding_server_error_retries: int = 2
    embedding_server_error_backoff_seconds: float = 1.0
    embedding_total_timeout_seconds: float = 180.0
    # 连续服务失败后短暂冷却，避免故障期间持续压垮上游。
    service_failure_threshold: int = 3
    service_cooldown_seconds: float = 30.0

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
    # P2 为不同决策上下文分配独立的基础召回配额。
    memory_context_top_k: dict[str, int] = field(default_factory=lambda: {
        "world": 5,
        "social": 4,
        "conversation": 3,
        "opinion_assessment": 6,
    })
    # 关闭后可用于无记忆消融；默认每次决策都先执行基础召回。
    memory_forced_recall_enabled: bool = True
    # 记忆检索超时后直接返回已有结果，避免 embedding/Chroma 慢调用拖住 tick。
    memory_retrieval_timeout_seconds: float = 30.0
    # 相同记忆查询在短时间内复用结果，减少相邻 tick 的重复 SQLite/Chroma 检索。
    memory_retrieval_ttl_ticks: int = 3
    # LLM planner 默认开启；基础召回后只补查仍然缺失的信息。
    memory_planner_enabled: bool = True
    # 当前观察足够行动时可跳过补查，但不会跳过 P2 基础召回。
    memory_planner_skip_when_observation_sufficient: bool = True
    # 人物档案默认使用规则摘要，避免每个 tick 产生大量 LLM 请求。
    memory_person_profile_llm_enabled: bool = False
    # 向量召回查询的 UTF-8 字节上限；不影响长期记忆正文写入。
    memory_semantic_query_max_bytes: int = 6000
    memory_prompt_max_chars_per_item: int = 1200
    memory_prompt_max_total_chars: int = 5000
    # 轻量记忆维护：只控制最近性、低价值事件和活跃向量数量。
    memory_forgetting_enabled: bool = True
    memory_maintenance_interval_ticks: int = 10
    memory_recency_window_ticks: int = 100
    memory_event_retention_ticks: int = 100
    memory_event_max_prunable_importance: float = 0.5
    memory_event_active_limit_per_agent: int = 2000
    memory_vector_active_limit_per_agent: int = 500
    memory_protected_importance: float = 0.7
    # 社交浏览只返回最近 N 条可见帖子；每条帖子仍保留全部评论。
    social_visible_post_limit: int = 10
    # 推荐系统开关：当前只保留配置入口，默认关闭，推荐算法暂不实现。
    social_recommendation_enabled: bool = False
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
    # relax 耗尽后仍允许低速移动，单次最多前进 5 格且不再扣减 relax。
    relax_zero_move_max_steps: int = 5
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
    # 正式消融总开关；关闭后不创建评测窗口，也不发起心理 LLM 请求。
    psychological_assessment_enabled: bool = True
    psychological_assessment_mode: str = "llm"
    psychological_llm_fallback_to_rule: bool = True
    # 兼容旧配置的总角色卡开关；两个细分开关用于隔离行为与观念评测路径。
    dynamic_role_card_enabled: bool = True
    dynamic_role_card_behavior_enabled: bool = True
    dynamic_role_card_opinion_enabled: bool = True

    # Social platform
    popularity_like_weight: float = 1.0
    popularity_repost_weight: float = 2.0
    popularity_comment_weight: float = 1.0
    default_online_trust: float = 0.5

    # Opinion assessment
    initial_opinion: float = OPINION_NEUTRAL
    default_opinion_topic: str = JIANG_PING_TOPIC
    opinion_assessment_history_limit: int = 200
    opinion_assessment_mode: str = "llm_as_judge"
    # llm_as_judge 在模拟中生成 current honest belief，并由本地 FLAN-T5-Large 评分。
    opinion_assessment_interval: int = 5
    opinion_flan_model_name: str = "google/flan-t5-large"
    # llm_voting 仅在模拟结束后执行，每个窗口固定覆盖十个时间步。
    opinion_voting_window_size: int = 10
    opinion_voter_count: int = 10
    # 结束后投票独立限峰，避免单账户持续并发触发上游限流和服务熔断。
    opinion_voting_max_concurrent_requests: int = 3
    opinion_voting_request_interval_seconds: float = 1.0
    # 没有线上发帖或评论的窗口直接记为不可评测，不发送无意义请求。
    opinion_voting_skip_empty_windows: bool = True
    # llm_as_judge 保留事件触发与间隔兜底行为。
    opinion_assessment_triggered_only: bool = True
    opinion_max_delta_per_assessment: float = 0.25
    opinion_assessment_recent_social: int = 5
    post_opinion_scoring_mode: str = "llm"
    post_opinion_llm_fallback_to_rule: bool = True

    # Micro-reflection
    micro_reflect_interval: int = 3
    # 任务完成后的轨迹反思默认交给 LLM 总结。
    trajectory_summary_llm_enabled: bool = True
