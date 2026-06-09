import asyncio
from dotenv import load_dotenv
from default_scenario import DEFAULT_AGENT_IDS, build_cli_demo_runtime
from persona.logger import setup_logging


async def _main():
    rt = build_cli_demo_runtime(conversation_max_rounds=2)
    agents = [rt.world.agents[agent_id] for agent_id in DEFAULT_AGENT_IDS]

    # --- 运行仿真 ---
    header = "  ".join(f"{ag.id:>8}" for ag in agents)
    print(f"\n=== 初始观念 ===\n  {'tick':>4}  {header}")
    print(f"  {'init':>4}  " + "  ".join(f"{ag.opinion:>8.3f}" for ag in agents))

    for tick in range(10):
        await rt.world.astep()
        marker = " ◀offline" if tick + 1 in (3, 6, 9) else ""
        print(f"  {tick+1:>4}  " + "  ".join(f"{ag.opinion:>8.3f}" for ag in agents) + marker)


if __name__ == "__main__":
    setup_logging()
    load_dotenv()
    asyncio.run(_main())


