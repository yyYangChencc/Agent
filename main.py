import os
from dotenv import load_dotenv
from persona.logger import setup_logging
from persona.runtime import SimulationRuntime
from world.objects import food

if __name__ == "__main__":
    setup_logging()
    load_dotenv()

    rt = SimulationRuntime.build(conversation_max_rounds=2)
    rt.mem.reset_all()

    # --- 创建智能体 ---
    a = rt.create_agent("agent_1", [3, 3],
        role="保守主义者，倾向于节约资源，不喜欢变化",
        speaking_style="沉稳、措辞谨慎")
    b = rt.create_agent("agent_2", [4, 2],
        role="积极探索者，乐于尝试新事物并分享经验",
        speaking_style="热情、喜欢分享")
    c = rt.create_agent("agent_3", [6, 6],
        role="中立观察者，善于倾听各方意见后再表态",
        speaking_style="理性、措辞中立")
    d = rt.create_agent("agent_4", [2, 8],
        role="激进改革派，主张打破现有秩序追求效率",
        speaking_style="直接、充满激情")
    e = rt.create_agent("agent_5", [8, 4],
        role="社区协调员，重视群体和谐与共识",
        speaking_style="温和、善于调解")

    agents = [a, b, c, d, e]

    # --- 设置初始观念 ---
    a.opinion = 0.15   # 极保守
    b.opinion = 0.45   # 略保守
    c.opinion = 0.50   # 中立
    d.opinion = 0.85   # 极激进
    e.opinion = 0.60   # 略激进

    # --- 好友关系（offline_trust >= 0.6 视为线下邻居）---
    # 阵营1：a、b、c 互相认识
    a.offline_trust["agent_2"] = 0.75
    a.offline_trust["agent_3"] = 0.65
    b.offline_trust["agent_1"] = 0.75
    b.offline_trust["agent_3"] = 0.70
    c.offline_trust["agent_1"] = 0.65
    c.offline_trust["agent_2"] = 0.70
    # 阵营2：d、e 互相认识
    d.offline_trust["agent_5"] = 0.80
    e.offline_trust["agent_4"] = 0.80
    # 跨阵营：c 与 e 是桥梁
    c.offline_trust["agent_5"] = 0.62
    e.offline_trust["agent_3"] = 0.62

    # --- 线上信任（默认 0.5，按关系调整）---
    a.online_trust["agent_2"] = 0.55
    a.online_trust["agent_3"] = 0.60
    b.online_trust["agent_1"] = 0.50
    b.online_trust["agent_4"] = 0.35   # b 不太信任激进派
    c.online_trust["agent_1"] = 0.55
    c.online_trust["agent_4"] = 0.55
    c.online_trust["agent_5"] = 0.60
    d.online_trust["agent_5"] = 0.65
    d.online_trust["agent_1"] = 0.30   # d 不信任保守派
    e.online_trust["agent_4"] = 0.60
    e.online_trust["agent_3"] = 0.65

    # --- 关注关系（能看到被关注者的帖子）---
    a.add_follower("agent_2"); a.add_follower("agent_3")
    b.add_follower("agent_1"); b.add_follower("agent_3"); b.add_follower("agent_5")
    c.add_follower("agent_1"); c.add_follower("agent_4"); c.add_follower("agent_5")
    d.add_follower("agent_5"); d.add_follower("agent_3")
    e.add_follower("agent_4"); e.add_follower("agent_3"); e.add_follower("agent_2")

    # --- 放置食物 ---
    food("food_1", 1, 2, [10, 10], rt.world)
    food("food_2", 1, 2, [5, 15], rt.world)
    food("food_3", 1, 2, [18, 5], rt.world)

    # --- 注入初始记忆 ---
    rt.mem.store_agent_memory("agent_1", "在（10，10）附近可能存在食物",
        memory_type="system", importance=0.9)
    rt.mem.store_agent_memory("agent_4", "在（5，15）附近可能存在食物",
        memory_type="system", importance=0.9)
    rt.mem.store_agent_memory("agent_5", "在（18，5）附近可能存在食物",
        memory_type="system", importance=0.9)

    # --- 运行仿真（offline 更新在 tick 3、6、9 触发）---
    header = "  ".join(f"{ag.id:>8}" for ag in agents)
    print(f"\n=== 初始观念 ===\n  {'tick':>4}  {header}")
    print(f"  {'init':>4}  " + "  ".join(f"{ag.opinion:>8.3f}" for ag in agents))

    for tick in range(10):
        rt.world.step()
        marker = " ◀offline" if tick + 1 in (3, 6, 9) else ""
        print(f"  {tick+1:>4}  " + "  ".join(f"{ag.opinion:>8.3f}" for ag in agents) + marker)


