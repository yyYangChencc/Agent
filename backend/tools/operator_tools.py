from tools.base import Tool
from social_sys.post.post import Post
from social_sys.post.comment import Comment
from world.objects import Interactable
from persona.opinion.scorer import evaluate_opinion
from persona.logger import get_logger

logger = get_logger(__name__)

def register_operator_tools(operator):
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
    lines.append("当你决定使用工具时，请严格输出如下 JSON，不要包含任何额外文本：")
    lines.append('{\n  "tool": "<工具名>",\n  "args": { "<参数名>": <参数值> }\n}')
    return tools, '\n'.join(lines)


class Operator:
    def __init__(self, world):
        self.world = world
        self.tool_specs = {
            "move": {
                "description": "自动向目标坐标 (x, y) 寻路移动，每次可前进多格，但不会超过观测半径大小，自动避开障碍物",
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
                "description": "在指定ID的床上休息，进入睡眠状态，经过 sleep_time 步后恢复 50 relax",
                "args": {"ID": str},
                "returns": str,
                "constraint": "目标必须是床(bed)，且在欧氏距离√2范围内（即相邻格子）"
            },
            "buy": {
                "description": "在指定ID的商店里进行购买",
                "args": {"ID": str},
                "returns": str,
                "constraint": "目标必须是食品店(food_shop)，且在欧氏距离√2范围内（即相邻格子）"
            },
            "enter_building": {
                "description": "进入指定ID的建筑内部，进入后智能体位置与建筑重合，前端不单独显示智能体",
                "args": {"ID": str},
                "returns": str,
                "constraint": "目标必须是建筑(building/bed/food_shop/playground/company)，且在欧氏距离√2范围内"
            },
            "exit_building": {
                "description": "离开当前所在建筑，回到建筑外部",
                "args": {},
                "returns": str,
                "constraint": "必须当前处于某建筑内部"
            }
        }

    def social_step(self, operator_ID):
        logger.info("[%s] 正在查看帖子...", operator_ID)
        agent = self.world.agents[operator_ID]
        return agent.social_step()

    def move(self, operator_ID: str, x: int, y: int):
        x = int(x)
        y = int(y)
        if operator_ID not in self.world.agents:
            return "智能体不存在"
        agent = self.world.agents[operator_ID]
        old_x, old_y = agent.position
        if x < 0 or y < 0 or x >= self.world.map.height or y >= self.world.map.width:
            return f"移动失败，[{x},{y}]超出地图边界"
        if x == old_x and y == old_y:
            return "目标位置是当前位置，没有发生移动"

        # Detect interactable object at target cell
        target_cell_id = self.world.map.get_e(x, y)
        target_obj = self.world.objects.get(target_cell_id) if target_cell_id != '0' else None
        target_interactable = isinstance(target_obj, Interactable)

        n = agent.config.observation_radius
        cur_x, cur_y = old_x, old_y
        steps = 0
        interact_msg = ""

        def _try_interact(cx, cy):
            nonlocal interact_msg
            if not target_interactable or interact_msg:
                return
            if target_cell_id not in self.world.objects:
                return
            if (cx - x) ** 2 + (cy - y) ** 2 > agent.config.eat_distance_sq:
                return
            with self.world._world_lock:
                if target_cell_id not in self.world.objects:
                    return
                result = target_obj.interact(agent)
            interact_msg = f"，并{result}"
            logger.info("[%s] 移动中交互 %s: %s", operator_ID, target_cell_id, result)

        # Check eat range at starting position before any movement
        _try_interact(cur_x, cur_y)

        for _ in range(n):
            if cur_x == x and cur_y == y:
                break
            best_pos = None
            best_dist = float("inf")
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cur_x + dx, cur_y + dy
                if nx < 0 or ny < 0 or nx >= self.world.map.height or ny >= self.world.map.width:
                    continue
                if not self.world.map.is_empty(nx, ny):
                    continue
                dist = abs(nx - x) + abs(ny - y)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = (nx, ny)
            if best_pos is None:
                break
            cur_x, cur_y = best_pos
            steps += 1
            # After each step, check if now in eat range of target object
            _try_interact(cur_x, cur_y)

        if steps == 0:
            if interact_msg:
                return f"{operator_ID}原地不动{interact_msg}"
            return f"无法移动，[{old_x},{old_y}]周围路径被阻挡"

        agent.update_satisfaction("relax", -agent.config.relax_moving_usage * steps)

        with self.world._world_lock:
            if not self.world.map.is_empty(cur_x, cur_y):
                return f"移动失败，[{cur_x},{cur_y}]已被占用，请下轮重试"
            self.world.map.place(cur_x, cur_y, agent.id)
            self.world.map.remove(old_x, old_y)
            agent.position = [cur_x, cur_y]

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
            kind = ID.split('_')[0]
            if kind != "food":
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
        logger.info("[%s] 开始睡觉 → %s", operator_ID, ID)
        return f"{operator_ID}{result}"

    def buy(self, operator_ID: str, ID: str):
        if ID == '0':
            return "此处为空"
        if operator_ID not in self.world.agents:
            return "智能体不存在"
        agent = self.world.agents[operator_ID]
        with self.world._world_lock:
            if ID not in self.world.objects:
                return "物品不存在"
            target = self.world.objects[ID]
            if getattr(target, "kind", None) != "food_shop":
                return "目标不是食品店，不可以购买"
            t_pos = target.get_position()
            a_pos = agent.get_position()
            if (t_pos[0] - a_pos[0]) ** 2 + (t_pos[1] - a_pos[1]) ** 2 > agent.config.eat_distance_sq:
                return "距离过远无法购买"
            result = target.interact(agent)
        logger.info("[%s] buy 占位调用 → %s", operator_ID, ID)
        return f"{operator_ID}{result}"

    def enter_building(self, operator_ID: str, ID: str):
        if ID == '0':
            return "此处为空"
        if operator_ID not in self.world.agents:
            return "智能体不存在"
        agent = self.world.agents[operator_ID]
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
            result = target.enter(agent)
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
            result = target.exit(agent)
        logger.info("[%s] 离开建筑 %s", operator_ID, building_id)
        return result


class SocialOperator:
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

    def send_message(self, operator_ID, content, ID):
        logger.info("[%s] 私信 → %s: %s", operator_ID, ID, content)
        return f"{operator_ID}对{ID}说:{content}"

    def send_post(self, operator_ID, content):
        with self.platform._posts_lock:
            post_id = len(self.platform.posts) + 1
            new_post = Post(post_id, operator_ID, content)
            new_post.opinion_index = evaluate_opinion(content)
            self.platform.posts.append(new_post)
        self.platform.get_agent(operator_ID).add_post_history(new_post)
        logger.info("[%s] 发表帖子: %s", operator_ID, content)
        return f"{operator_ID}成功发表了帖子: {content}"

    def comment_post(self, operator_ID, post_id, content):
        post = next((p for p in self.platform.posts if p.id == post_id), None)
        if not post:
            logger.warning("[%s] 评论失败，帖子 %s 不存在", operator_ID, post_id)
            return "帖子不存在"
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
        post = next((p for p in self.platform.posts if p.id == post_id), None)
        if not post:
            logger.warning("[%s] 点赞失败，帖子 %s 不存在", operator_ID, post_id)
            return "帖子不存在"
        post.add_like(operator_ID)
        agent = self.platform.get_agent(operator_ID)
        if agent is not None and post.author_id != operator_ID:
            cfg = agent.config
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
        post = next((p for p in self.platform.posts if p.id == post_id), None)
        if not post:
            logger.warning("[%s] 点踩失败，帖子 %s 不存在", operator_ID, post_id)
            return "帖子不存在"
        post.add_dislike(operator_ID)
        agent = self.platform.get_agent(operator_ID)
        if agent is not None and post.author_id != operator_ID:
            cfg = agent.config
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
