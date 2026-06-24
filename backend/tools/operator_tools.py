import heapq

from tools.base import Tool
from social_sys.post.post import Post
from social_sys.post.comment import Comment
from world.objects import Interactable, building
from persona.opinion.scorer import evaluate_opinion
from persona.logger import get_logger

logger = get_logger(__name__)

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
    lines.append("工具调用 JSON 的形状如下：")
    lines.append('{\n  "tool": "<工具名>",\n  "args": { "<参数名>": <参数值> }\n}')
    return tools, '\n'.join(lines)


class Operator:
    """线下世界工具集合。

    LLM 只输出工具 JSON；World.execute 再调用这里的方法改变地图、需求和对象状态。
    """

    def __init__(self, world):
        self.world = world
        self.tool_specs = {
            "move": {
                "description": "自动向目标坐标 (x, y) 使用网格最短路寻路移动，每次可前进多格，但不会超过观测半径大小，自动避开障碍物并优先利用道路",
                "args": {"x": int, "y": int},
                "returns": str,
                "constraint": "若目标坐标(x,y)存在障碍物，将停在距(x,y)最近的可达位置；若周围路径全被阻挡则原地不动"
            },
            "eat": {
                "description": "吃指定ID的食物",
                "args": {"ID": str},
                "returns": str,
                "constraint": "食物必须在欧氏距离√2范围内（即相邻格子），且不能与食物站在同一坐标上"
            },
            "speak": {
                "description": "对指定ID或<all>说话,说话内容为content,若对多个ID说话则用空格分隔ID，response_to表示回复的内容，若此条语句不是回复他人的发言，则可无需添加response_to.",
                "args": {"content": str, "ID": str, "response_to": str},
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
                "constraint": "目标必须是床(bed)，且在欧氏距离√2范围内（即相邻格子）"
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

    def social_step(self, operator_ID):
        logger.info("[%s] 正在查看帖子...", operator_ID)
        agent = self.world.agents[operator_ID]
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

        max_steps = max(0, int(agent.config.observation_radius))
        interact_msg = ""

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
            path = planned_path[:max_steps + 1]
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
                _try_interact_at(cur_x, cur_y)

        if steps == 0:
            if interact_msg:
                return f"{operator_ID}原地不动{interact_msg}"
            return f"无法移动，[{old_x},{old_y}]已是当前可达范围内距目标最近的位置"

        agent.update_satisfaction("relax", -agent.config.relax_moving_usage * steps)

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
        if operator_ID not in self.world.agents:
            return "智能体不存在"
        agent = self.world.agents[operator_ID]
        with self.world._world_lock:
            if ID not in self.world.objects:
                logger.warning("[%s] 物品 %s 不存在", operator_ID, ID)
                return "物品不存在"
            operated = self.world.objects[ID]
            if getattr(operated, "kind", None) != "food":
                logger.warning("[%s] 物品 %s 种类不是食物", operator_ID, ID)
                return "物品种类不是食物，不可以吃"
            f_pos = operated.get_position()
            A_pos = agent.get_position()
            if (f_pos[0] - A_pos[0])**2 + (f_pos[1] - A_pos[1])**2 > agent.config.eat_distance_sq:
                logger.warning("[%s] 距离食物 %s 过远", operator_ID, ID)
                return "距离过远吃不到"
            operated.interact(agent)
        logger.info("[%s] 成功吃到 %s", operator_ID, ID)
        return f"{operator_ID}成功吃到{ID}"

    def speak(self, operator_ID, content, ID, response_to=None):
        msg = f"{operator_ID}对{ID}说:{content}"
        if response_to:
            msg += f"  回复：{response_to}"
        logger.info("[%s] 说话 → %s | 内容: %s", operator_ID, ID, content)
        return msg

    def sleep(self, operator_ID: str, ID: str):
        if ID == '0':
            return "此处为空"
        if operator_ID not in self.world.agents:
            return "智能体不存在"
        agent = self.world.agents[operator_ID]
        if agent.sleeping:
            return "已经在睡眠中"
        if agent.inside_building_id:
            return "当前在建筑内，需先离开建筑再睡觉"
        with self.world._world_lock:
            if ID not in self.world.objects:
                return "物品不存在"
            target = self.world.objects[ID]
            if getattr(target, "kind", None) != "bed":
                return "目标不是床，不可以睡觉"
            t_pos = target.get_position()
            a_pos = agent.get_position()
            if (t_pos[0] - a_pos[0]) ** 2 + (t_pos[1] - a_pos[1]) ** 2 > agent.config.eat_distance_sq:
                return "距离过远无法休息"
            result = target.interact(agent)
        if agent.sleeping:
            logger.info("[%s] 开始睡觉 → %s", operator_ID, ID)
        return f"{operator_ID}{result}"

    def enter_building(self, operator_ID: str, ID: str):
        if ID == '0':
            return "此处为空"
        if operator_ID not in self.world.agents:
            return "智能体不存在"
        agent = self.world.agents[operator_ID]
        if agent.sleeping:
            return "睡眠中无法进入建筑"
        if agent.inside_building_id:
            if agent.inside_building_id == ID:
                return f"已在建筑 {ID} 内"
            return "当前已在其他建筑内，请先离开建筑"
        with self.world._world_lock:
            if ID not in self.world.objects:
                return "物品不存在"
            target = self.world.objects[ID]
            from world.objects import building
            if not isinstance(target, building):
                return "目标不是建筑，无法进入"
            t_pos = target.get_position()
            a_pos = agent.get_position()
            if (t_pos[0] - a_pos[0]) ** 2 + (t_pos[1] - a_pos[1]) ** 2 > agent.config.eat_distance_sq:
                return "距离过远无法进入"
            old_x, old_y = agent.position
            enter_result = target.enter(agent)
            if self._in_bounds(old_x, old_y) and self.world.map.get_e(old_x, old_y) == agent.id:
                self.world.map.remove(old_x, old_y)
            agent.position = list(target.position)
        interact_result = self.world.interact_inside_building(agent, record_history=False, source="enter_building")
        result = f"{enter_result}; {interact_result}" if interact_result else enter_result
        logger.info("[%s] 进入建筑 %s", operator_ID, ID)
        return result

    def exit_building(self, operator_ID: str):
        if operator_ID not in self.world.agents:
            return "智能体不存在"
        agent = self.world.agents[operator_ID]
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
        logger.info("[%s] 离开建筑 %s", operator_ID, building_id)
        return result


class SocialOperator:
    """线上社交平台工具集合。

    这里强制所有互动只能作用于当前 social_step 返回给智能体的帖子，避免 LLM
    编造 post_id 或操作不可见帖子。
    """

    def __init__(self, platform):
        self.platform = platform
        self.tool_specs = {
            "send_post": {
                "description": "发表帖子，content为帖子内容",
                "args": {"content": str},
                "returns": str,
            },
            "comment_post": {
                "description": "评论帖子，post_id为被评论的帖子ID，content为评论内容",
                "args": {"post_id": int, "content": str},
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
            }
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

    def send_message(self, operator_ID, content, ID):
        logger.info("[%s] 私信 → %s: %s", operator_ID, ID, content)
        return f"{operator_ID}对{ID}说:{content}"

    def send_post(self, operator_ID, content):
        # 发帖时的 opinion_index 目前由 evaluate_opinion 占位函数给出。
        with self.platform._posts_lock:
            post_id = len(self.platform.posts) + 1
            new_post = Post(post_id, operator_ID, content)
            new_post.opinion_index = evaluate_opinion(content)
            self.platform.posts.append(new_post)
        self.platform.get_agent(operator_ID).add_post_history(new_post)
        logger.info("[%s] 发表帖子: %s", operator_ID, content)
        return f"{operator_ID}成功发表了帖子: {content}"

    def comment_post(self, operator_ID, post_id, content):
        post, post_id, error = self._resolve_visible_post(operator_ID, post_id)
        if error:
            return error
        comment_id = f"{post_id}_c{post.comments+1}"
        new_comment = Comment(comment_id, operator_ID, content, time=None)
        post.add_comment(new_comment)
        # 通知帖主有人评论了其帖子
        if post.author_id != operator_ID:
            author_agent = self.platform.get_agent(post.author_id)
            if author_agent is not None:
                snippet = content[:40] + "..." if len(content) > 40 else content
                author_agent._pending_social_notifications.append(
                    f"[社交通知] {operator_ID} 评论了你的帖子：{snippet}"
                )
        logger.info("[%s] 评论帖子 %s: %s", operator_ID, post_id, content)
        return f"{operator_ID}成功评论了帖子 {post_id}: {content}"

    def like_post(self, operator_ID, post_id):
        post, post_id, error = self._resolve_visible_post(operator_ID, post_id)
        if error:
            return error
        post.add_like(operator_ID)
        agent = self.platform.get_agent(operator_ID)
        if agent is not None and post.author_id != operator_ID:
            cfg = agent.config
            # 点赞被视作弱正反馈，只调整 online_trust，不直接更新观念分数。
            agent.online_trust[post.author_id] = min(
                1.0,
                agent.online_trust.get(post.author_id, cfg.default_online_trust) + 0.05
            )
            # 通知帖主有人点赞
            author_agent = self.platform.get_agent(post.author_id)
            if author_agent is not None:
                author_agent._pending_social_notifications.append(
                    f"[社交通知] {operator_ID} 点赞了你的帖子"
                )
        logger.info("[%s] 点赞帖子 %s", operator_ID, post_id)
        return f"{operator_ID}成功点赞了帖子 {post_id}"

    def dislike_post(self, operator_ID, post_id):
        post, post_id, error = self._resolve_visible_post(operator_ID, post_id)
        if error:
            return error
        post.add_dislike(operator_ID)
        agent = self.platform.get_agent(operator_ID)
        if agent is not None and post.author_id != operator_ID:
            cfg = agent.config
            # 点踩被视作弱负反馈，只调整 online_trust，不直接更新观念分数。
            agent.online_trust[post.author_id] = max(
                0.0,
                agent.online_trust.get(post.author_id, cfg.default_online_trust) - 0.05
            )
            # 通知帖主有人点踩
            author_agent = self.platform.get_agent(post.author_id)
            if author_agent is not None:
                author_agent._pending_social_notifications.append(
                    f"[社交通知] {operator_ID} 点踩了你的帖子"
                )
        logger.info("[%s] 点踩帖子 %s", operator_ID, post_id)
        return f"{operator_ID}成功点踩了帖子 {post_id}"
