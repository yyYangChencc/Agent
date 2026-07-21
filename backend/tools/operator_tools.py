import heapq
import math

from social_sys.post import Comment, Post
from world.objects import Interactable, building
from persona.need_events import apply_need_delta
from persona.logger import get_logger

logger = get_logger(__name__)


class Tool:
    """工具函数的最小包装，统一保留名称、描述和参数声明。"""

    def __init__(self, name, func, description, args):
        self.name = name
        self.func = func
        self.description = description
        self.args = args

    def run(self, **kwargs):
        return self.func(**kwargs)


def register_operator_tools(operator):
    """把 Operator 方法包装成 Tool，并生成给 LLM 使用的工具说明文本。"""

    tool_specs = operator.tool_specs
    tools = {}
    for name, spec in tool_specs.items():
        func = getattr(operator, name, None)
        if func is None:
            raise ValueError(f"operator 未实现方法: {name}")
        tools[name] = Tool(
            name=name,
            func=func,
            description=spec["description"],
            args=spec["args"]
        )
    lines = []
    lines.append("你是一个智能体，可以使用以下工具与环境交互：\n")
    for i, (name, spec) in enumerate(tool_specs.items(), start=1):
        lines.append(f"{i}. 工具名：{name}")
        lines.append(f"   功能：{spec['description']}")
        lines.append("   参数：")
        for arg, arg_type in spec["args"].items():
            lines.append(f"     - {arg} ({arg_type.__name__})")
        lines.append(f"   返回：{spec['returns'].__name__}")
        if "constraint" in spec:
            lines.append(f"   约束：{spec['constraint']}")
        lines.append("")
    lines.append("外层决策 JSON 的 action 字段中，工具调用 JSON 的形状如下：")
    lines.append('{\n  "tool": "<工具名>",\n  "args": { "<参数名>": <参数值> }\n}')
    return tools, '\n'.join(lines)


class Operator:
    """线下世界工具集合。

    LLM 输出包含 think 和 action 的决策 JSON；World.execute 取出 action 后调用这里的方法改变地图、需求和对象状态。
    """

    def __init__(self, world):
        self.world = world
        self.tool_specs = {
            "move": {
                "description": "自动向目标坐标 (x, y) 使用网格最短路寻路移动，单次移动不再按观测半径封顶；relax 大于 0 时每格消耗 relax，relax 为 0 时按低速上限移动且不再扣减 relax",
                "args": {"x": int, "y": int},
                "returns": str,
                "constraint": "若目标坐标(x,y)存在障碍物，将停在距(x,y)最近的可达位置；relax 为 0 时每次最多移动 5 格"
            },
            "eat": {
                "description": "吃指定ID的食物",
                "args": {"ID": str},
                "returns": str,
                "constraint": "食物必须在欧氏距离√2范围内（即相邻格子），且不能与食物站在同一坐标上"
            },
            "speak": {
                "description": "对指定ID或<all>说话,说话内容为content,若对多个ID说话则用空格分隔ID，response_to表示回复的内容，若此条语句不是回复他人的发言，则可无需添加response_to.",
                "args": {
                    "content": str,
                    "ID": str,
                    "response_to": str,
                    "intent": str,
                    "social_valence": float,
                    "topic": str,
                    "topic_stance": float,
                },
                "returns": str,
                "constraint": "距离己方5格内才能听见"
            },
            "social_step": {
                "description": "浏览社交平台或发布内容。适合分享经历、表达观点、了解他人动态。如果你已知目标位置，应使用 move 前往，而非发帖询问",
                "args": {},
                "returns": str,
            },
            "sleep": {
                "description": "在指定ID的床上休息，并进入睡眠状态；睡眠期间每个时间步恢复一部分放松度，睡眠结束前不会行动",
                "args": {"ID": str},
                "returns": str,
                "constraint": "目标必须是床(bed)，且在欧氏距离√2范围内（即相邻格子）；目标带 owner_agent_id 时只能由该智能体使用"
            },
            "enter_building": {
                "description": "进入指定ID的建筑内部；进入后会立即自动触发该建筑的 interact 效果，之后每个时间步若仍在建筑内也会自动触发 interact",
                "args": {"ID": str},
                "returns": str,
                "constraint": "目标必须是建筑(building/food_shop/playground/company)，且在欧氏距离√2范围内"
            },
            "exit_building": {
                "description": "离开当前所在建筑，回到建筑外部",
                "args": {},
                "returns": str,
                "constraint": "必须当前处于某建筑内部"
            }
        }

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.world.map.height and 0 <= y < self.world.map.width

    def _adjacent_empty_cell(self, position: list[int]) -> list[int] | None:
        bx, by = position
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx, ny = bx + dx, by + dy
            if self._in_bounds(nx, ny) and self.world.map.is_empty(nx, ny):
                return [nx, ny]
        return None

    def _agent_or_error(self, operator_ID: str):
        """统一读取实体智能体，避免各工具重复判断。"""

        agent = self.world.agents.get(operator_ID)
        if agent is None:
            return None, "智能体不存在"
        return agent, None

    def _object_or_error(self, ID: str, *, empty_message: str = "此处为空"):
        """统一读取地图物体，保留原工具的失败返回语义。"""

        if ID == '0':
            return None, empty_message
        obj = self.world.objects.get(ID)
        if obj is None:
            return None, "物品不存在"
        return obj, None

    def _within_interact_distance(self, agent, target) -> bool:
        """判断智能体是否处于通用交互距离内。"""

        target_pos = target.get_position()
        agent_pos = agent.get_position()
        return (
            (target_pos[0] - agent_pos[0]) ** 2
            + (target_pos[1] - agent_pos[1]) ** 2
        ) <= agent.config.eat_distance_sq

    @staticmethod
    def _remember_tool(agent, tool_name: str) -> None:
        """记录最近使用的工具，供需求与历史链路读取。"""

        if hasattr(agent, "remember_action_tool"):
            agent.remember_action_tool(tool_name)

    def social_step(self, operator_ID):
        logger.info("[%s] 正在查看帖子...", operator_ID)
        agent = self.world.agents[operator_ID]
        self._remember_tool(agent, "social_step")
        return agent.social_step()

    @staticmethod
    def _distance_sq(a: tuple[int, int], b: tuple[int, int]) -> int:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _road_cells(self) -> set[tuple[int, int]]:
        """读取地图设计中的道路格；道路在寻路中有更低移动成本。"""

        design = getattr(self.world, "map_design", None)
        if not isinstance(design, dict):
            return set()
        roads = design.get("roads", [])
        cells: set[tuple[int, int]] = set()
        for road in roads:
            for row, col in road.get("cells", []):
                if 0 <= row < self.world.map.height and 0 <= col < self.world.map.width:
                    cells.add((row, col))
        return cells

    def _is_walkable_cell(self, pos: tuple[int, int], start: tuple[int, int]) -> bool:
        row, col = pos
        if row < 0 or col < 0 or row >= self.world.map.height or col >= self.world.map.width:
            return False
        return pos == start or self.world.map.is_empty(row, col)

    def _ordered_neighbors(self, pos: tuple[int, int], target: tuple[int, int]) -> list[tuple[int, int]]:
        row, col = pos
        neighbors = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ]
        neighbors.sort(key=lambda cell: (self._manhattan(cell, target), cell[0], cell[1]))
        return neighbors

    def _movement_cost(self, pos: tuple[int, int], road_cells: set[tuple[int, int]]) -> float:
        return 0.7 if pos in road_cells else 1.0

    def _reconstruct_path(
        self,
        parents: dict[tuple[int, int], tuple[int, int] | None],
        goal: tuple[int, int],
    ) -> list[list[int]]:
        path: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = goal
        while cur is not None:
            path.append(cur)
            cur = parents[cur]
        path.reverse()
        return [[row, col] for row, col in path]

    def _find_path(
        self,
        start: tuple[int, int],
        target: tuple[int, int],
        *,
        stop_within_distance_sq: float | None = None,
    ) -> list[list[int]]:
        """在当前网格上规划路径。

        除移动者起点外，占用格视为障碍。如果目标格被占用或不可达，
        路径会停在最接近目标的可达格；若目标是可交互物品，可停在交互距离内。
        """

        road_cells = self._road_cells()
        distances: dict[tuple[int, int], float] = {start: 0.0}
        parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        heap: list[tuple[float, int, int, int]] = [(0.0, start[0], start[1], 0)]

        while heap:
            cost, row, col, _ = heapq.heappop(heap)
            pos = (row, col)
            if cost > distances[pos]:
                continue
            for next_pos in self._ordered_neighbors(pos, target):
                if not self._is_walkable_cell(next_pos, start):
                    continue
                next_cost = cost + self._movement_cost(next_pos, road_cells)
                if next_cost >= distances.get(next_pos, float("inf")):
                    continue
                distances[next_pos] = next_cost
                parents[next_pos] = pos
                heapq.heappush(heap, (next_cost, next_pos[0], next_pos[1], len(parents)))

        if stop_within_distance_sq is not None:
            # 对食物等可交互目标，只要走到交互半径内即可，不要求踩到目标格。
            reachable_goals = [
                pos for pos in distances
                if self._distance_sq(pos, target) <= stop_within_distance_sq
            ]
            if reachable_goals:
                goal = min(
                    reachable_goals,
                    key=lambda pos: (
                        distances[pos],
                        self._distance_sq(pos, target),
                        self._manhattan(pos, target),
                        pos[0],
                        pos[1],
                    ),
                )
                return self._reconstruct_path(parents, goal)

        if target in distances:
            return self._reconstruct_path(parents, target)

        goal = min(
            distances,
            key=lambda pos: (
                self._distance_sq(pos, target),
                self._manhattan(pos, target),
                distances[pos],
                pos[0],
                pos[1],
            ),
        )
        return self._reconstruct_path(parents, goal)

    def move(self, operator_ID: str, x: int, y: int):
        x = int(x)
        y = int(y)
        if operator_ID not in self.world.agents:
            return "智能体不存在"
        agent = self.world.agents[operator_ID]
        if x < 0 or y < 0 or x >= self.world.map.height or y >= self.world.map.width:
            return f"移动失败，[{x},{y}]超出地图边界"

        exit_msg = ""
        if agent.inside_building_id:
            # 移动前必须先离开建筑，否则地图坐标和 inside_building_id 会不一致。
            exit_msg = self.exit_building(operator_ID)
            if agent.inside_building_id:
                return exit_msg

        old_x, old_y = agent.position
        if x == old_x and y == old_y:
            if exit_msg:
                return f"{exit_msg}；目标位置是当前位置，没有发生移动"
            return "目标位置是当前位置，没有发生移动"

        interact_msg = ""
        relax_cost = max(0.0, float(agent.config.relax_moving_usage))
        current_relax = max(0.0, float(agent.satisfaction.get("relax", 0.0)))

        with self.world._world_lock:
            target_cell_id = self.world.map.get_e(x, y)
            target_obj = self.world.objects.get(target_cell_id) if target_cell_id != '0' else None
            target_interactable = isinstance(target_obj, Interactable) and not isinstance(target_obj, building)
            stop_distance_sq = agent.config.eat_distance_sq if target_interactable else None

            # 路径规划在锁内完成，避免并发 agent 同时占用同一个目标格。
            planned_path = self._find_path(
                (old_x, old_y),
                (x, y),
                stop_within_distance_sq=stop_distance_sq,
            )
            if current_relax <= 0 and len(planned_path) > 1:
                # relax 耗尽后仍可低速前进，避免远离床铺时永久卡住。
                zero_relax_max_steps = max(1, int(agent.config.relax_zero_move_max_steps))
                path = planned_path[:zero_relax_max_steps + 1]
            elif relax_cost > 0:
                # relax 只在每个格子移动完成后扣除；剩余 relax 大于 0 时允许再走一格。
                max_steps_by_relax = max(1, int(math.ceil(current_relax / relax_cost)))
                path = planned_path[:max_steps_by_relax + 1]
            else:
                path = planned_path
            if not path:
                path = [[old_x, old_y]]
            cur_x, cur_y = path[-1]
            steps = max(0, len(path) - 1)

            def _try_interact_at(cx: int, cy: int) -> None:
                """移动过程中到达交互范围时，顺手触发目标物品的 interact。"""

                nonlocal interact_msg
                if not target_interactable or interact_msg:
                    return
                if target_cell_id not in self.world.objects:
                    return
                if (cx - x) ** 2 + (cy - y) ** 2 > agent.config.eat_distance_sq:
                    return
                result = target_obj.interact(agent)
                interact_msg = f"，并{result}"
                logger.info("[%s] 移动中交互 %s: %s", operator_ID, target_cell_id, result)

            if steps == 0:
                _try_interact_at(old_x, old_y)
            else:
                if not self.world.map.is_empty(cur_x, cur_y):
                    return f"移动失败，[{cur_x},{cur_y}]已被占用，请下轮重试"
                if self.world.map.get_e(old_x, old_y) == agent.id:
                    self.world.map.remove(old_x, old_y)
                self.world.map.place(cur_x, cur_y, agent.id)
                agent.position = [cur_x, cur_y]
                self.world.movements.append({
                    "agent_id": operator_ID,
                    "path": path,
                })
                agent.did_move_this_tick = True
                _try_interact_at(cur_x, cur_y)

        if steps == 0:
            if interact_msg:
                return f"{operator_ID}原地不动{interact_msg}"
            return f"无法移动，[{old_x},{old_y}]已是当前可达范围内距目标最近的位置"

        if current_relax > 0:
            apply_need_delta(
                agent,
                "relax",
                -agent.config.relax_moving_usage * steps,
                source="physiological",
                reason="移动消耗 relax",
                evidence={
                    "from": [old_x, old_y],
                    "to": [cur_x, cur_y],
                    "target": [x, y],
                    "steps": steps,
                },
            )
        self._remember_tool(agent, "move")

        logger.info(
            "[%s] 从 [%d,%d] 移动 %d 步至 [%d,%d]（目标 [%d,%d]）",
            operator_ID, old_x, old_y, steps, cur_x, cur_y, x, y,
        )
        if cur_x == x and cur_y == y:
            return f"{operator_ID}成功到达目标[{x},{y}]（共移动{steps}步）{interact_msg}"
        return f"{operator_ID}移动{steps}步至[{cur_x},{cur_y}]，目标[{x},{y}]尚未到达{interact_msg}"

    def eat(self, operator_ID: str, ID: str):
        if ID == '0':
            logger.warning("[%s] 尝试吃空位置", operator_ID)
            return "此处为空"
        agent, error = self._agent_or_error(operator_ID)
        if error:
            return error
        with self.world._world_lock:
            operated, error = self._object_or_error(ID)
            if error:
                logger.warning("[%s] 物品 %s 不存在", operator_ID, ID)
                return error
            if getattr(operated, "kind", None) != "food":
                logger.warning("[%s] 物品 %s 种类不是食物", operator_ID, ID)
                return "物品种类不是食物，不可以吃"
            if not self._within_interact_distance(agent, operated):
                logger.warning("[%s] 距离食物 %s 过远", operator_ID, ID)
                return "距离过远吃不到"
            operated.interact(agent)
        self._remember_tool(agent, "eat")
        logger.info("[%s] 成功吃到 %s", operator_ID, ID)
        return f"{operator_ID}成功吃到{ID}"

    def speak(
        self,
        operator_ID,
        content,
        ID,
        response_to=None,
        intent=None,
        social_valence=None,
        topic="",
        topic_stance=None,
    ):
        msg = f"{operator_ID}对{ID}说:{content}"
        if response_to:
            msg += f"  回复：{response_to}"
        logger.info("[%s] 说话 → %s | 内容: %s", operator_ID, ID, content)
        # speak 不因元数据非法而失败；规范化由 ConversationManager 统一处理。
        if intent:
            msg += f" intent={intent}"
        if social_valence is not None:
            msg += f" social_valence={social_valence}"
        if topic:
            msg += f" topic={topic}"
        if topic_stance is not None:
            msg += f" topic_stance={topic_stance}"
        return msg

    def sleep(self, operator_ID: str, ID: str):
        if ID == '0':
            return "此处为空"
        agent, error = self._agent_or_error(operator_ID)
        if error:
            return error
        if agent.sleeping:
            return "已经在睡眠中"
        if agent.inside_building_id:
            return "当前在建筑内，需先离开建筑再睡觉"
        with self.world._world_lock:
            target, error = self._object_or_error(ID)
            if error:
                return error
            if getattr(target, "kind", None) != "bed":
                return "目标不是床，不可以睡觉"
            if hasattr(target, "can_be_used_by") and not target.can_be_used_by(agent.id):
                return f"床 {target.id} 是 {target.owner_agent_id} 的专属床铺，{agent.id} 无法使用"
            if not self._within_interact_distance(agent, target):
                return "距离过远无法休息"
            result = target.interact(agent)
        if agent.sleeping:
            self._remember_tool(agent, "sleep")
            logger.info("[%s] 开始睡觉 → %s", operator_ID, ID)
        return f"{operator_ID}{result}"

    def enter_building(self, operator_ID: str, ID: str):
        if ID == '0':
            return "此处为空"
        agent, error = self._agent_or_error(operator_ID)
        if error:
            return error
        if agent.sleeping:
            return "睡眠中无法进入建筑"
        if agent.inside_building_id:
            if agent.inside_building_id == ID:
                return f"已在建筑 {ID} 内"
            return "当前已在其他建筑内，请先离开建筑"
        with self.world._world_lock:
            target, error = self._object_or_error(ID)
            if error:
                return error
            if not isinstance(target, building):
                return "目标不是建筑，无法进入"
            if not self._within_interact_distance(agent, target):
                return "距离过远无法进入"
            old_x, old_y = agent.position
            enter_result = target.enter(agent)
            if self._in_bounds(old_x, old_y) and self.world.map.get_e(old_x, old_y) == agent.id:
                self.world.map.remove(old_x, old_y)
            agent.position = list(target.position)
        interact_result = self.world.interact_inside_building(agent, record_history=False, source="enter_building")
        self._remember_tool(agent, "enter_building")
        self._apply_building_exploration_needs(agent, target)
        result = f"{enter_result}; {interact_result}" if interact_result else enter_result
        logger.info("[%s] 进入建筑 %s", operator_ID, ID)
        return result

    def exit_building(self, operator_ID: str):
        agent, error = self._agent_or_error(operator_ID)
        if error:
            return error
        building_id = agent.inside_building_id
        if not building_id:
            return "当前不在任何建筑内"
        with self.world._world_lock:
            if building_id not in self.world.objects:
                agent.inside_building_id = None
                return "建筑不存在，已强制退出"
            target = self.world.objects[building_id]
            exit_pos = self._adjacent_empty_cell(target.position)
            if exit_pos is None:
                return "建筑周围没有空位，无法离开"
            result = target.exit(agent)
            old_x, old_y = agent.position
            if self._in_bounds(old_x, old_y) and self.world.map.get_e(old_x, old_y) == agent.id:
                self.world.map.remove(old_x, old_y)
            self.world.map.place(exit_pos[0], exit_pos[1], agent.id)
            agent.position = exit_pos
        self._remember_tool(agent, "exit_building")
        logger.info("[%s] 离开建筑 %s", operator_ID, building_id)
        return result

    def _apply_building_exploration_needs(self, agent, target) -> None:
        """首次进入建筑或建筑类型时，补充自我实现满足度。"""

        building_id = str(getattr(target, "id", "") or "")
        building_kind = str(getattr(target, "kind", "") or "")
        gained_exploration = False
        if building_id and building_id not in agent.visited_building_ids:
            agent.visited_building_ids.add(building_id)
            event = apply_need_delta(
                agent,
                "self_actualization",
                agent.config.self_actualization_first_building_delta,
                source="exploration",
                reason="首次进入新的建筑",
                evidence={"building_id": building_id, "kind": building_kind},
            )
            gained_exploration = gained_exploration or event is not None
        if building_kind and building_kind not in agent.visited_building_kinds:
            agent.visited_building_kinds.add(building_kind)
            event = apply_need_delta(
                agent,
                "self_actualization",
                agent.config.self_actualization_first_building_kind_delta,
                source="exploration",
                reason="首次进入新的建筑类型",
                evidence={"building_id": building_id, "kind": building_kind},
            )
            gained_exploration = gained_exploration or event is not None
        if gained_exploration:
            self._apply_self_actualization_task_bonus(
                agent,
                "完成自我实现任务时产生新的地点探索",
                {"building_id": building_id, "kind": building_kind},
            )

    def _apply_self_actualization_task_bonus(self, agent, reason: str, evidence: dict) -> None:
        """自我实现任务产生新探索或新表达时给予额外反馈。"""

        if getattr(agent, "task_urgency_key", "") != "self_actualization":
            return
        apply_need_delta(
            agent,
            "self_actualization",
            agent.config.self_actualization_task_completion_delta,
            source="exploration",
            reason=reason,
            evidence=evidence,
        )


class SocialOperator:
    """线上社交平台工具集合。

    帖子、评论和账号目标均受本轮浏览载荷约束，避免 LLM 编造标识符或操作
    未展示的对象。
    """

    def __init__(self, platform):
        self.platform = platform
        self.tool_specs = {
            "send_post": {
                "description": "发表帖子，topic为帖子标题/主题，content为帖子内容，opinion_index为作者对该主题的立场快照，范围[-1,1]；仅在有自己的新增观点、真实经历或明确求助时使用，不用于复述他人帖子；若内容与当前观念主题相关，topic必须使用该观念主题，否则自行定义普通主题",
                "args": {"topic": str, "content": str, "opinion_index": float},
                "returns": str,
            },
            "comment_post": {
                "description": "评论帖子，post_id为被评论的帖子ID，content为评论内容，agreement_to_post为你对被评论帖子的认同程度，范围[-1,1]",
                "args": {"post_id": int, "content": str, "agreement_to_post": float},
                "returns": str,
            },
            "like_post": {
                "description": "点赞帖子，post_id为被点赞的帖子ID",
                "args": {"post_id": int},
                "returns": str,
            },
            "dislike_post": {
                "description": "点踩帖子，post_id为被点踩的帖子ID",
                "args": {"post_id": int},
                "returns": str,
            },
            "reply_comment": {
                "description": "回复当前可见帖子中的评论，post_id和comment_id必须来自本轮浏览结果，agreement_to_post仍表示对原帖的认同程度",
                "args": {"post_id": int, "comment_id": str, "content": str, "agreement_to_post": float},
                "returns": str,
            },
            "follow_author": {
                "description": "关注账号，author_id必须逐字取自本轮浏览结果的account_ids",
                "args": {"author_id": str},
                "returns": str,
            },
            "unfollow_author": {
                "description": "取消关注账号，author_id必须逐字取自本轮浏览结果的following_ids",
                "args": {"author_id": str},
                "returns": str,
            },
            "repost_post": {
                "description": "转发当前可见帖子，opinion_index为转发者对该主题的立场快照，范围[-1,1]",
                "args": {"post_id": int, "opinion_index": float},
                "returns": str,
            },
            "quote_post": {
                "description": "引用当前可见帖子并添加自己的内容，opinion_index为引用者对该主题的立场快照，范围[-1,1]",
                "args": {"post_id": int, "content": str, "opinion_index": float},
                "returns": str,
            },
        }

    def _visible_posts_for(self, operator_ID: str):
        agent = self.platform.get_agent(operator_ID)
        if agent is None:
            return []
        return list(getattr(agent, "_last_seen_posts", []) or [])

    def _resolve_visible_post(self, operator_ID: str, post_id):
        """校验 post_id 是否是当前可互动帖子列表中的整数 ID。"""

        if isinstance(post_id, bool):
            logger.warning("[%s] 社交动作失败，post_id 非整数: %r", operator_ID, post_id)
            return None, None, "post_id 必须是当前可互动帖子ID列表中的整数"
        if isinstance(post_id, int):
            normalized_post_id = post_id
        elif isinstance(post_id, str) and post_id.strip().isdigit():
            normalized_post_id = int(post_id.strip())
        else:
            logger.warning("[%s] 社交动作失败，post_id 非整数: %r", operator_ID, post_id)
            return None, None, "post_id 必须是当前可互动帖子ID列表中的整数"

        visible_posts = self._visible_posts_for(operator_ID)
        visible_ids = [post.id for post in visible_posts]
        post = next((post for post in visible_posts if post.id == normalized_post_id), None)
        if post is None:
            visible_text = "、".join(str(pid) for pid in visible_ids) if visible_ids else "无"
            logger.warning(
                "[%s] 社交动作失败，帖子 %s 不在当前可互动帖子ID列表中；可互动ID: %s",
                operator_ID,
                normalized_post_id,
                visible_text,
            )
            return None, normalized_post_id, f"帖子{normalized_post_id}不在当前可互动帖子ID列表中；可互动ID：{visible_text}"
        return post, normalized_post_id, None

    def _resolve_visible_comment(self, operator_ID: str, post_id, comment_id):
        """只允许回复当前可见帖子中精确列出的评论 ID。"""

        post, normalized_post_id, error = self._resolve_visible_post(operator_ID, post_id)
        if error:
            return None, None, normalized_post_id, error
        if not isinstance(comment_id, str) or not comment_id:
            return None, post, normalized_post_id, "comment_id 必须是当前可见帖子中的非空字符串"
        agent = self.platform.get_agent(operator_ID)
        visible_comment_ids = (
            getattr(agent, "_last_seen_comment_ids_by_post", {}).get(normalized_post_id, set())
            if agent is not None
            else set()
        )
        if comment_id not in visible_comment_ids:
            return None, post, normalized_post_id, f"评论{comment_id}不在帖子{normalized_post_id}的本轮展示评论列表中"
        comment = post.get_comment(comment_id) if hasattr(post, "get_comment") else None
        if comment is None:
            return None, post, normalized_post_id, f"评论{comment_id}已不在帖子{normalized_post_id}中"
        return comment, post, normalized_post_id, None

    def _resolve_account(self, operator_ID: str, author_id, *, require_followed: bool = False):
        """按本轮账号目录或当前关注列表校验精确账号 ID。"""

        agent = self.platform.get_agent(operator_ID)
        if agent is None:
            return None, None, "operator_ID 未注册为平台智能体"
        if not isinstance(author_id, str) or not author_id:
            return agent, None, "author_id 必须是非空字符串"
        allowed = list(agent.followers) if require_followed else list(getattr(agent, "_last_browse_account_ids", []) or [])
        if author_id not in allowed:
            return agent, author_id, f"author_id={author_id} 不在当前允许列表中"
        if author_id not in self.platform.agents and author_id not in self.platform.influencers:
            return agent, author_id, f"author_id={author_id} 未注册"
        return agent, author_id, None

    def _record_platform_event(
        self,
        event_type: str,
        *,
        operator_ID: str,
        post_id=None,
        target_agent_id: str = "",
        details: dict | None = None,
    ) -> dict:
        """把成功社交动作写入统一平台事件账本。"""

        agent = self.platform.get_agent(operator_ID)
        event = self.platform.record_event(
            event_type,
            actor_id=operator_ID,
            post_id=post_id,
            target_agent_id=target_agent_id,
            feed_request_id=str(getattr(agent, "_last_feed_request_id", "") or "") if agent is not None else "",
            details=details,
        )
        if agent is not None:
            social_action = getattr(agent, "last_social_action", {}) or {}
            if social_action.get("action") == event_type:
                social_action["platform_event_id"] = event["event_id"]
                social_action["feed_request_id"] = event["feed_request_id"]
        return event

    def send_message(self, operator_ID, content, ID):
        logger.info("[%s] 私信 → %s: %s", operator_ID, ID, content)
        return f"{operator_ID}对{ID}说:{content}"

    def _parse_required_unit_score(self, operator_ID: str, field_name: str, value):
        """校验 LLM 显式给出的 [-1, 1] 数值字段，非法时拒绝动作。"""

        if value is None or isinstance(value, bool):
            logger.warning("[%s] 社交动作失败，%s 缺失或不是数值: %r", operator_ID, field_name, value)
            return None, f"error: {field_name} 必须是 [-1, 1] 范围内的数值"
        try:
            number = float(value)
        except (TypeError, ValueError):
            logger.warning("[%s] 社交动作失败，%s 不是数值: %r", operator_ID, field_name, value)
            return None, f"error: {field_name} 必须是 [-1, 1] 范围内的数值"
        if not math.isfinite(number) or number < -1.0 or number > 1.0:
            logger.warning("[%s] 社交动作失败，%s 超出范围: %r", operator_ID, field_name, value)
            return None, f"error: {field_name} 必须在 [-1, 1] 范围内"
        return number, None

    @staticmethod
    def _remember_social_tool(agent, tool_name: str) -> None:
        """记录线上工具动作，供历史和需求事件读取。"""

        if hasattr(agent, "remember_action_tool"):
            agent.remember_action_tool(tool_name)

    def _set_last_social_action(
        self,
        agent,
        *,
        action: str,
        post_id="",
        post_content: str = "",
        comment_content: str = "",
        opinion_index=0.0,
        agreement_to_post="",
        post_topic: str = "",
        state_changed: bool = True,
        previous_reaction="",
        current_reaction="",
        source_post_id=None,
        root_post_id=None,
        source_author_id: str = "",
        target_agent_id: str = "",
        comment_id: str = "",
        parent_comment_id: str = "",
        root_comment_id: str = "",
    ) -> None:
        """统一写入最近一次线上动作快照，保持历史字段一致。"""

        agent.last_social_action = {
            "action": action,
            "post_id": post_id,
            "post_content": post_content,
            "comment_content": comment_content,
            "opinion_index": opinion_index,
            "agreement_to_post": agreement_to_post,
            "post_topic": post_topic,
            "state_changed": bool(state_changed),
            "previous_reaction": previous_reaction,
            "current_reaction": current_reaction,
            "source_post_id": source_post_id,
            "root_post_id": root_post_id,
            "source_author_id": source_author_id,
            "target_agent_id": target_agent_id,
            "comment_id": comment_id,
            "parent_comment_id": parent_comment_id,
            "root_comment_id": root_comment_id,
        }

    @staticmethod
    def _event_evidence(platform_event: dict) -> dict:
        """提取需求事件使用的平台关联字段。"""

        return {
            "platform_event_id": platform_event.get("event_id", ""),
            "feed_request_id": platform_event.get("feed_request_id", ""),
        }

    def _notification_payload(
        self,
        target_agent,
        message: str,
        platform_event: dict,
        *,
        post: Post | None = None,
    ) -> dict:
        """把平台事件投影为可持久化的结构化通知。"""

        details = dict(platform_event.get("details") or {})
        actor = self.platform.get_agent(platform_event.get("actor_id"))
        episode_id = str(getattr(actor, "_current_episode_id", "") or "") if actor is not None else ""
        if not episode_id:
            episode_id = str(getattr(target_agent, "_current_episode_id", "") or "")
        source_author_id = details.get("source_author_id")
        if source_author_id in (None, "") and post is not None:
            source_author_id = post.author_id
        root_post_id = details.get("root_post_id")
        if root_post_id is None and post is not None:
            root_post_id = post.root_post_id
        actor_id = str(platform_event.get("actor_id") or "")
        return {
            "schema_version": 1,
            "type": "notification",
            "content": message,
            "time": int(platform_event.get("tick") or self.platform.time),
            "episode_id": episode_id,
            "event_id": str(platform_event.get("event_id") or ""),
            "event_type": str(platform_event.get("event_type") or ""),
            "platform_event_id": str(platform_event.get("event_id") or ""),
            "feed_request_id": str(platform_event.get("feed_request_id") or ""),
            "actor_id": actor_id,
            "related_agent_id": actor_id or str(source_author_id or ""),
            "post_id": platform_event.get("post_id"),
            "comment_id": details.get("comment_id"),
            "parent_comment_id": details.get("parent_comment_id"),
            "root_comment_id": details.get("root_comment_id"),
            "target_agent_id": target_agent.id,
            "source_post_id": details.get("source_post_id"),
            "root_post_id": root_post_id,
            "source_author_id": source_author_id,
            "topic": getattr(post, "topic", "") if post is not None else "",
            "details": details,
        }

    def _notify_author(self, post: Post, message: str, platform_event: dict) -> None:
        """向真实帖子作者写入社交通知；无实体投放者会被跳过。"""

        author_agent = self.platform.get_agent(post.author_id)
        if author_agent is not None:
            author_agent._pending_social_notifications.append(
                self._notification_payload(author_agent, message, platform_event, post=post)
            )

    def _adjust_online_trust(self, agent, author_id: str, delta: float) -> tuple[float, float]:
        """线上弱反馈只调整信任，不直接修改观念。"""

        current = agent.online_trust.get(author_id, agent.config.default_online_trust)
        updated = max(0.0, min(1.0, current + delta))
        agent.online_trust[author_id] = updated
        return current, updated

    def _after_reaction_to_post(
        self,
        *,
        operator_ID: str,
        agent,
        post: Post,
        action: str,
        trust_delta: float,
        notification: str,
        belonging_delta: float,
        esteem_delta: float,
        platform_event: dict,
    ) -> None:
        """统一处理点赞/点踩后的信任、通知、需求和新关系事件。"""

        if agent is None or post.author_id == operator_ID:
            return
        trust_before, trust_after = self._adjust_online_trust(agent, post.author_id, trust_delta)
        social_action = getattr(agent, "last_social_action", {}) or {}
        if social_action.get("action") == action:
            social_action["online_trust_before"] = trust_before
            social_action["online_trust_after"] = trust_after
        self._notify_author(post, notification, platform_event)
        self._apply_author_feedback(
            operator_ID,
            post,
            action,
            belonging_delta,
            esteem_delta,
            platform_event=platform_event,
        )
        self._apply_new_social_contact(agent, post, action, platform_event)

    @staticmethod
    def _reaction_effect_deltas(agent, previous, current) -> tuple[float, float, float]:
        """按反应状态迁移计算信任、归属和尊重的净变化。"""

        trust_values = {None: 0.0, "like": 0.05, "dislike": -0.05}
        belonging_values = {
            None: 0.0,
            "like": agent.config.social_like_belonging_delta,
            "dislike": agent.config.social_dislike_belonging_delta,
        }
        esteem_values = {
            None: 0.0,
            "like": agent.config.social_like_esteem_delta,
            "dislike": agent.config.social_dislike_esteem_delta,
        }
        return (
            trust_values[current] - trust_values[previous],
            belonging_values[current] - belonging_values[previous],
            esteem_values[current] - esteem_values[previous],
        )

    def send_post(self, operator_ID, content, topic="日常", opinion_index=None):
        topic = str(topic or "").strip() or "日常"
        score, error = self._parse_required_unit_score(operator_ID, "opinion_index", opinion_index)
        if error:
            return error
        # 普通智能体发帖立场由动作 LLM 显式给出，工具层只校验和记录。
        with self.platform._posts_lock:
            post_id = self.platform.allocate_post_id()
            new_post = Post(post_id, operator_ID, content, topic=topic)
            new_post.opinion_index = score
            new_post.time = self.platform.time
            self.platform.add_post(new_post)
        agent = self.platform.get_agent(operator_ID)
        if agent is not None:
            agent.add_post_history(new_post)
            self._remember_social_tool(agent, "send_post")
            self._set_last_social_action(
                agent,
                action="send_post",
                post_id=new_post.id,
                post_content=content,
                opinion_index=new_post.opinion_index,
                post_topic=topic,
            )
        platform_event = self._record_platform_event(
            "send_post",
            operator_ID=operator_ID,
            post_id=new_post.id,
            details={
                "content": new_post.content,
                "topic": topic,
                "opinion_index": score,
                "is_news": False,
                "is_rumor": bool(new_post.is_rumor),
                "source_type": new_post.source_type,
            },
        )
        if agent is not None:
            self._apply_topic_expression_need(agent, topic, new_post, platform_event)
        logger.info("[%s] 发表帖子 topic=%s: %s", operator_ID, topic, content)
        return f"{operator_ID}成功发表了帖子，主题：{topic}，内容：{content}"

    def comment_post(self, operator_ID, post_id, content, agreement_to_post=None):
        post, post_id, error = self._resolve_visible_post(operator_ID, post_id)
        if error:
            return f"error: {error}"
        agreement, error = self._parse_required_unit_score(operator_ID, "agreement_to_post", agreement_to_post)
        if error:
            return error
        comment_id = f"{post_id}_c{post.comments+1}"
        new_comment = Comment(
            comment_id,
            operator_ID,
            content,
            time=self.platform.time,
            agreement_to_post=agreement,
        )
        new_comment = post.add_comment(new_comment)
        agent = self.platform.get_agent(operator_ID)
        if agent is not None:
            self._remember_social_tool(agent, "comment_post")
            self._set_last_social_action(
                agent,
                action="comment_post",
                post_id=post_id,
                comment_content=content,
                opinion_index=getattr(post, "opinion_index", 0.0),
                agreement_to_post=agreement,
                post_topic=getattr(post, "topic", ""),
                root_post_id=getattr(post, "root_post_id", None),
                source_author_id=str(post.author_id or ""),
                target_agent_id=str(post.author_id or ""),
                comment_id=new_comment.id,
                root_comment_id=new_comment.root_comment_id,
            )
        platform_event = self._record_platform_event(
            "comment_post",
            operator_ID=operator_ID,
            post_id=post_id,
            target_agent_id=str(post.author_id or ""),
            details={
                "comment_id": new_comment.id,
                "content": new_comment.content,
                "agreement_to_post": agreement,
                "parent_comment_id": new_comment.parent_comment_id,
                "root_comment_id": new_comment.root_comment_id,
            },
        )
        if agent is not None:
            self._apply_new_social_contact(agent, post, "comment_post", platform_event)
        # 通知帖主有人评论了其帖子
        if post.author_id != operator_ID:
            snippet = content[:40] + "..." if len(content) > 40 else content
            self._notify_author(
                post,
                f"[社交通知] {operator_ID} 评论了你的帖子：{snippet}",
                platform_event,
            )
            self._apply_comment_feedback_to_author(
                operator_ID,
                post,
                agreement,
                action="comment_post",
                platform_event=platform_event,
            )
        logger.info("[%s] 评论帖子 %s: %s", operator_ID, post_id, content)
        return f"{operator_ID}成功评论了帖子 {post_id}: {content}"

    def like_post(self, operator_ID, post_id):
        post, post_id, error = self._resolve_visible_post(operator_ID, post_id)
        if error:
            return f"error: {error}"
        reaction = post.add_like(operator_ID)
        agent = self.platform.get_agent(operator_ID)
        if agent is not None:
            self._remember_social_tool(agent, "like_post")
            self._set_last_social_action(
                agent,
                action="like_post",
                post_id=post_id,
                post_content=getattr(post, "content", ""),
                post_topic=getattr(post, "topic", ""),
                opinion_index=getattr(post, "opinion_index", 0.0),
                state_changed=reaction["changed"],
                previous_reaction=reaction["previous"],
                current_reaction=reaction["current"],
                root_post_id=getattr(post, "root_post_id", None),
                source_author_id=str(post.author_id or ""),
                target_agent_id=str(post.author_id or ""),
            )
        platform_event = self._record_platform_event(
            "like_post",
            operator_ID=operator_ID,
            post_id=post_id,
            target_agent_id=str(post.author_id or ""),
            details={
                "changed": reaction["changed"],
                "previous": reaction["previous"],
                "current": reaction["current"],
                "likes": post.likes,
                "dislikes": post.dislikes,
            },
        )
        if agent is not None and reaction["changed"]:
            trust_delta, belonging_delta, esteem_delta = self._reaction_effect_deltas(
                agent,
                reaction["previous"],
                reaction["current"],
            )
            self._after_reaction_to_post(
                operator_ID=operator_ID,
                agent=agent,
                post=post,
                action="like_post",
                trust_delta=trust_delta,
                notification=f"[社交通知] {operator_ID} 点赞了你的帖子",
                belonging_delta=belonging_delta,
                esteem_delta=esteem_delta,
                platform_event=platform_event,
            )
        logger.info("[%s] 点赞帖子 %s", operator_ID, post_id)
        return f"{operator_ID}成功点赞了帖子 {post_id}"

    def dislike_post(self, operator_ID, post_id):
        post, post_id, error = self._resolve_visible_post(operator_ID, post_id)
        if error:
            return f"error: {error}"
        reaction = post.add_dislike(operator_ID)
        agent = self.platform.get_agent(operator_ID)
        if agent is not None:
            self._remember_social_tool(agent, "dislike_post")
            self._set_last_social_action(
                agent,
                action="dislike_post",
                post_id=post_id,
                post_content=getattr(post, "content", ""),
                post_topic=getattr(post, "topic", ""),
                opinion_index=getattr(post, "opinion_index", 0.0),
                state_changed=reaction["changed"],
                previous_reaction=reaction["previous"],
                current_reaction=reaction["current"],
                root_post_id=getattr(post, "root_post_id", None),
                source_author_id=str(post.author_id or ""),
                target_agent_id=str(post.author_id or ""),
            )
        platform_event = self._record_platform_event(
            "dislike_post",
            operator_ID=operator_ID,
            post_id=post_id,
            target_agent_id=str(post.author_id or ""),
            details={
                "changed": reaction["changed"],
                "previous": reaction["previous"],
                "current": reaction["current"],
                "likes": post.likes,
                "dislikes": post.dislikes,
            },
        )
        if agent is not None and reaction["changed"]:
            trust_delta, belonging_delta, esteem_delta = self._reaction_effect_deltas(
                agent,
                reaction["previous"],
                reaction["current"],
            )
            self._after_reaction_to_post(
                operator_ID=operator_ID,
                agent=agent,
                post=post,
                action="dislike_post",
                trust_delta=trust_delta,
                notification=f"[社交通知] {operator_ID} 点踩了你的帖子",
                belonging_delta=belonging_delta,
                esteem_delta=esteem_delta,
                platform_event=platform_event,
            )
        logger.info("[%s] 点踩帖子 %s", operator_ID, post_id)
        return f"{operator_ID}成功点踩了帖子 {post_id}"

    def reply_comment(self, operator_ID, post_id, comment_id, content, agreement_to_post=None):
        parent, post, post_id, error = self._resolve_visible_comment(operator_ID, post_id, comment_id)
        if error:
            return f"error: {error}"
        agreement, error = self._parse_required_unit_score(operator_ID, "agreement_to_post", agreement_to_post)
        if error:
            return error
        new_comment = Comment(
            f"{post_id}_c{post.comments + 1}",
            operator_ID,
            content,
            time=self.platform.time,
            agreement_to_post=agreement,
            parent_comment_id=parent.id,
            root_comment_id=parent.root_comment_id or parent.id,
        )
        new_comment = post.add_comment(new_comment)
        agent = self.platform.get_agent(operator_ID)
        if agent is not None:
            self._remember_social_tool(agent, "reply_comment")
            self._set_last_social_action(
                agent,
                action="reply_comment",
                post_id=post_id,
                comment_content=content,
                opinion_index=getattr(post, "opinion_index", 0.0),
                agreement_to_post=agreement,
                post_topic=getattr(post, "topic", ""),
                root_post_id=getattr(post, "root_post_id", None),
                source_author_id=str(post.author_id or ""),
                target_agent_id=str(parent.author_id or ""),
                comment_id=new_comment.id,
                parent_comment_id=parent.id,
                root_comment_id=new_comment.root_comment_id,
            )
        platform_event = self._record_platform_event(
            "reply_comment",
            operator_ID=operator_ID,
            post_id=post_id,
            target_agent_id=str(parent.author_id or ""),
            details={
                "comment_id": new_comment.id,
                "content": new_comment.content,
                "parent_comment_id": parent.id,
                "root_comment_id": new_comment.root_comment_id,
                "agreement_to_post": agreement,
            },
        )
        if agent is not None:
            self._apply_new_social_contact(agent, post, "reply_comment", platform_event)
        parent_author = self.platform.get_agent(parent.author_id)
        if parent_author is not None and parent.author_id != operator_ID:
            parent_author._pending_social_notifications.append(
                self._notification_payload(
                    parent_author,
                    f"[社交通知] {operator_ID} 回复了你在帖子 {post_id} 下的评论 {parent.id}：{content[:40]}",
                    platform_event,
                    post=post,
                )
            )
        if post.author_id != operator_ID:
            self._apply_comment_feedback_to_author(
                operator_ID,
                post,
                agreement,
                action="reply_comment",
                platform_event=platform_event,
            )
        logger.info("[%s] 回复评论 %s: %s", operator_ID, parent.id, content)
        return f"{operator_ID}成功回复了评论 {parent.id}: {content}"

    def follow_author(self, operator_ID, author_id):
        agent, author_id, error = self._resolve_account(operator_ID, author_id)
        if error:
            return f"error: {error}"
        changed = author_id not in agent.followers
        if changed:
            agent.add_follower(author_id)
        self._remember_social_tool(agent, "follow_author")
        self._set_last_social_action(
            agent,
            action="follow_author",
            target_agent_id=author_id,
            state_changed=changed,
        )
        self._record_platform_event(
            "follow_author",
            operator_ID=operator_ID,
            target_agent_id=author_id,
            details={"changed": changed},
        )
        return f"{operator_ID}已关注账号 {author_id}" if changed else f"{operator_ID}已经关注账号 {author_id}"

    def unfollow_author(self, operator_ID, author_id):
        agent, author_id, error = self._resolve_account(operator_ID, author_id, require_followed=True)
        if error:
            return f"error: {error}"
        agent.followers.remove(author_id)
        self._remember_social_tool(agent, "unfollow_author")
        self._set_last_social_action(
            agent,
            action="unfollow_author",
            target_agent_id=author_id,
            state_changed=True,
        )
        self._record_platform_event(
            "unfollow_author",
            operator_ID=operator_ID,
            target_agent_id=author_id,
            details={"changed": True},
        )
        return f"{operator_ID}已取消关注账号 {author_id}"

    def repost_post(self, operator_ID, post_id, opinion_index=None):
        post, post_id, error = self._resolve_visible_post(operator_ID, post_id)
        if error:
            return f"error: {error}"
        score, error = self._parse_required_unit_score(operator_ID, "opinion_index", opinion_index)
        if error:
            return error
        return self._create_repost(
            operator_ID=operator_ID,
            source_post=post,
            source_post_id=post_id,
            content=post.content,
            opinion_index=score,
            action="repost_post",
            source_type="repost",
        )

    def quote_post(self, operator_ID, post_id, content, opinion_index=None):
        post, post_id, error = self._resolve_visible_post(operator_ID, post_id)
        if error:
            return f"error: {error}"
        score, error = self._parse_required_unit_score(operator_ID, "opinion_index", opinion_index)
        if error:
            return error
        return self._create_repost(
            operator_ID=operator_ID,
            source_post=post,
            source_post_id=post_id,
            content=content,
            opinion_index=score,
            action="quote_post",
            source_type="quote_post",
        )

    def _create_repost(
        self,
        *,
        operator_ID: str,
        source_post: Post,
        source_post_id: int,
        content: str,
        opinion_index: float,
        action: str,
        source_type: str,
    ):
        """创建一次去重的转发或引用转发。"""

        repost_state = source_post.add_repost(operator_ID)
        if not repost_state["changed"]:
            return f"error: {operator_ID}已经转发或引用过帖子 {source_post_id}"
        post_id = self.platform.allocate_post_id()
        root_post_id = source_post.root_post_id or source_post.id
        new_post = Post(
            post_id,
            operator_ID,
            content,
            is_rumor=source_post.is_rumor,
            topic=source_post.topic,
            source_type=source_type,
            repost_of_post_id=source_post.id,
            root_post_id=root_post_id,
            source_author_id=source_post.author_id,
        )
        new_post.opinion_index = opinion_index
        new_post.time = self.platform.time
        self.platform.add_post(new_post)
        agent = self.platform.get_agent(operator_ID)
        if agent is not None:
            agent.add_post_history(new_post)
            self._remember_social_tool(agent, action)
            self._set_last_social_action(
                agent,
                action=action,
                post_id=new_post.id,
                post_content=content,
                opinion_index=opinion_index,
                post_topic=new_post.topic,
                source_post_id=source_post_id,
                root_post_id=root_post_id,
                source_author_id=str(source_post.author_id or ""),
                target_agent_id=str(source_post.author_id or ""),
            )
        platform_event = self._record_platform_event(
            action,
            operator_ID=operator_ID,
            post_id=new_post.id,
            target_agent_id=str(source_post.author_id or ""),
            details={
                "source_post_id": source_post_id,
                "root_post_id": root_post_id,
                "source_author_id": source_post.author_id,
                "content": new_post.content,
                "topic": new_post.topic,
                "source_type": source_type,
                "opinion_index": opinion_index,
                "is_news": bool(new_post.is_news),
                "is_rumor": bool(new_post.is_rumor),
            },
        )
        if agent is not None and action == "quote_post":
            self._apply_topic_expression_need(agent, new_post.topic, new_post, platform_event)
        if source_post.author_id != operator_ID:
            action_text = "引用了" if action == "quote_post" else "转发了"
            self._notify_author(
                source_post,
                f"[社交通知] {operator_ID} {action_text}你的帖子 {source_post_id}",
                platform_event,
            )
        logger.info("[%s] %s 帖子 %s -> %s", operator_ID, action, source_post_id, new_post.id)
        return f"{operator_ID}成功{('引用' if action == 'quote_post' else '转发')}帖子 {source_post_id}"

    def _apply_topic_expression_need(self, agent, topic: str, post: Post, platform_event: dict) -> None:
        """首次围绕系统新闻主题自主表达时，提高自我实现。"""

        default_topic = str(getattr(agent.config, "default_opinion_topic", "") or "")
        if not default_topic or str(topic or "") != default_topic:
            return
        if default_topic in agent.expressed_opinion_topics:
            return
        agent.expressed_opinion_topics.add(default_topic)
        evidence = {"post_id": post.id, "topic": topic, "opinion_index": post.opinion_index}
        evidence.update(self._event_evidence(platform_event))
        event = apply_need_delta(
            agent,
            "self_actualization",
            agent.config.self_actualization_first_topic_post_delta,
            source="exploration",
            reason="首次围绕系统新闻主题自主表达观点",
            evidence=evidence,
        )
        if event is not None:
            bonus_evidence = {"post_id": post.id, "topic": topic}
            bonus_evidence.update(self._event_evidence(platform_event))
            self._apply_self_actualization_task_bonus(
                agent,
                "完成自我实现任务时产生新的观点表达",
                bonus_evidence,
            )

    def _apply_comment_feedback_to_author(
        self,
        operator_ID: str,
        post: Post,
        agreement: float,
        *,
        action: str,
        platform_event: dict,
    ) -> None:
        """根据评论认同值影响原帖作者的归属和尊重。"""

        author_agent = self.platform.get_agent(post.author_id)
        if author_agent is None:
            return
        threshold = author_agent.config.social_feedback_agreement_threshold
        if agreement > threshold:
            self._apply_author_feedback(
                operator_ID,
                post,
                action,
                author_agent.config.social_comment_positive_belonging_delta,
                author_agent.config.social_comment_positive_esteem_delta,
                agreement_to_post=agreement,
                platform_event=platform_event,
            )
        elif agreement < -threshold:
            self._apply_author_feedback(
                operator_ID,
                post,
                action,
                author_agent.config.social_comment_negative_belonging_delta,
                author_agent.config.social_comment_negative_esteem_delta,
                agreement_to_post=agreement,
                platform_event=platform_event,
            )

    def _apply_author_feedback(
        self,
        operator_ID: str,
        post: Post,
        action: str,
        belonging_delta: float,
        esteem_delta: float,
        *,
        agreement_to_post: float | None = None,
        platform_event: dict,
    ) -> None:
        """把线上互动转成帖子作者的归属和尊重事件。"""

        author_agent = self.platform.get_agent(post.author_id)
        if author_agent is None:
            return
        evidence = {
            "post_id": post.id,
            "post_author_id": post.author_id,
            "operator_id": operator_ID,
            "action": action,
            "topic": getattr(post, "topic", ""),
            "opinion_index": getattr(post, "opinion_index", 0.0),
        }
        evidence.update(self._event_evidence(platform_event))
        if agreement_to_post is not None:
            evidence["agreement_to_post"] = agreement_to_post
        apply_need_delta(
            author_agent,
            "belonging",
            belonging_delta,
            source="social_feedback",
            reason="自己的帖子收到线上互动反馈",
            evidence=evidence,
        )
        apply_need_delta(
            author_agent,
            "esteem",
            esteem_delta,
            source="social_feedback",
            reason="自己的帖子收到线上互动反馈",
            evidence=evidence,
        )

    def _apply_new_social_contact(self, agent, post: Post, action: str, platform_event: dict) -> None:
        """自我实现任务中与新的真实智能体发生线上互动时给出反馈。"""

        author_id = str(getattr(post, "author_id", "") or "")
        if not author_id or author_id == agent.id:
            return
        if self.platform.get_agent(author_id) is None:
            return
        if author_id in agent.known_social_contacts:
            return
        agent.known_social_contacts.add(author_id)
        evidence = {"post_id": post.id, "author_id": author_id, "action": action}
        evidence.update(self._event_evidence(platform_event))
        self._apply_self_actualization_task_bonus(
            agent,
            "完成自我实现任务时产生新的线上互动对象",
            evidence,
        )

    def _apply_self_actualization_task_bonus(self, agent, reason: str, evidence: dict) -> None:
        """当前任务为自我实现且发生新表达或新互动时追加反馈。"""

        if getattr(agent, "task_urgency_key", "") != "self_actualization":
            return
        apply_need_delta(
            agent,
            "self_actualization",
            agent.config.self_actualization_task_completion_delta,
            source="exploration",
            reason=reason,
            evidence=evidence,
        )
