from __future__ import annotations
from typing import TYPE_CHECKING
from persona.logger import get_logger

if TYPE_CHECKING:
    from persona.agents.agent import Agent
    from persona.config import AgentConfig

logger = get_logger(__name__)


class OpinionUpdater:
    def __init__(self, config: "AgentConfig"):
        self.config = config

    def online_update(self, agent: "Agent") -> None:
        """Online opinion update: called after every tick."""
        posts = agent._last_seen_posts
        agent._last_seen_posts = []
        if not posts:
            return
        c = self.config
        trust_sum = sum(
            agent.online_trust.get(p.author_id, c.default_online_trust)
            for p in posts
        )
        if not trust_sum:
            return
        social_avg = sum(
            agent.online_trust.get(p.author_id, c.default_online_trust) * p.opinion_index
            for p in posts
        ) / trust_sum
        # Δ = α × (social_avg − opinion) × (1 − σ)
        delta = c.online_opinion_lr * (social_avg - agent.opinion) * (1 - c.self_confidence)
        agent.update_opinion(delta)
        logger.debug("[%s] 线上观念更新: %.3f → %.3f (social_avg=%.3f)",
                     agent.id, agent.opinion - delta, agent.opinion, social_avg)

    def offline_update(self, agent: "Agent", all_agents: dict) -> None:
        """Offline opinion + urgency conformity update: called every m ticks.
        Offline neighbors = agents whose offline_trust >= friend_trust_threshold.
        """
        c = self.config
        neighbors = [
            a for a in all_agents.values()
            if a.id != agent.id
            and agent.offline_trust.get(a.id, c.default_offline_trust) >= c.friend_trust_threshold
        ]
        if not neighbors:
            return

        # Opinion conformity
        t_sum = sum(agent.offline_trust.get(n.id, c.default_offline_trust) for n in neighbors)
        if t_sum:
            neighbor_avg = sum(
                agent.offline_trust.get(n.id, c.default_offline_trust) * n.opinion
                for n in neighbors
            ) / t_sum
            agent.update_opinion(c.offline_opinion_lr * (neighbor_avg - agent.opinion))
            logger.debug("[%s] 线下观念更新 → %.3f (邻居均值=%.3f, 邻居数=%d)",
                         agent.id, agent.opinion, neighbor_avg, len(neighbors))

        # Urgency conformity (urgency U updated via offline conformity method)
        for urgency_key in list(agent.urgency):
            dt_sum = sum(agent.offline_trust.get(n.id, c.default_offline_trust) for n in neighbors)
            if not dt_sum:
                continue
            avg_d = sum(
                agent.offline_trust.get(n.id, c.default_offline_trust) * n.urgency.get(urgency_key, 0)
                for n in neighbors
            ) / dt_sum
            delta_d = c.offline_opinion_lr * (avg_d - agent.urgency.get(urgency_key, 0))
            agent.update_urgency(urgency_key, delta_d)
            logger.debug("[%s] 线下需求更新 %s: delta=%.3f (邻居均值=%.3f)",
                         agent.id, urgency_key, delta_d, avg_d)
