class objects:
    """地图对象基类。

    对象创建时会自动注册到 world.objects 和地图；子类通过 kind 区分前端展示和交互逻辑。
    """

    def __init__(self, id, num, position, world):
        self.id = id
        self.num = num
        self.position = position
        self.world = world
        self.world.add_object(self)
        self.kind = "objects"
    def show_num(self):
        return self.num

    def get_position(self):
        return self.position

    def get_desc(self) -> str:
        return f"ID: {self.id}，类别: {self.kind}，数量 {self.num}"


class Interactable(objects):
    """可交互对象基类。"""

    def interact(self, agent) -> str:
        raise NotImplementedError

class Uninteractable(objects):
    """不可交互对象基类。"""


class food(Interactable):
    def __init__(self, id: str, num: int, provide: int, position: list, world):
        super().__init__(id, num, position, world)
        self.provide = provide
        self.kind = "food"

    def get_desc(self) -> str:
        return f"ID: {self.id}，类别：{self.kind}，功能：提供 {self.provide} 饱腹感，剩余数量 {self.num}"
        
    def interact(self, agent) -> str:
        # 食物是一次性地图物品，交互后减少库存，库存归零则从地图和对象表移除。
        agent.update_satisfaction("satiety", self.provide)
        self.eaten()
        return f"吃到了{self.id}"

    def eaten(self):
        self.num -= 1
        if self.num <= 0:
            self.world.map.remove(*self.position)
            self.world.objects.pop(self.id)


class building(Interactable):
    """场景中的建筑，作为可交互物品占据地图格位。进入建筑后由世界循环自动触发 interact。"""

    def __init__(self, id: str, position: list, world):
        # 建筑无数量属性，num 固定为 None（前端按"无限"渲染，不会因 num<=0 而隐藏）
        super().__init__(id, None, position, world)
        self.kind = "building"
        self.occupants: list[str] = []

    def interact(self, agent) -> str:
        # 建筑交互通过 world.auto_interact_inside_buildings 每 tick 自动触发。
        return f"与建筑 {self.id} 互动（暂未实现具体效果）"

    def enter(self, agent) -> str:
        if agent.id not in self.occupants:
            self.occupants.append(agent.id)
        agent.inside_building_id = self.id
        return f"{agent.id} 进入了 {self.id}"

    def exit(self, agent) -> str:
        if agent.id in self.occupants:
            self.occupants.remove(agent.id)
        agent.inside_building_id = None
        return f"{agent.id} 离开了 {self.id}"

    def get_desc(self) -> str:
        return f"ID: {self.id}，类别: {self.kind}"
    
    
class company(building):
    """公司：建筑子类，智能体进入并停留时自动获得工作效果。"""

    def __init__(self, id: str, position: list, world, salary=10, relax_cost=10):
        super().__init__(id, position, world)
        self.kind = "company"
        self.salary = salary  # 工作获得的工资
        self.relax_cost = relax_cost

    def interact(self, agent) -> str:
        # 公司是 money 获取渠道，同时用 relax 成本约束无限刷钱。
        agent.update_satisfaction("money", self.salary)
        agent.update_satisfaction("relax", -self.relax_cost)
        return f"在公司 {self.id} 工作，获得 {self.salary} 元工资，消耗 {self.relax_cost} relax"
    
    def get_desc(self) -> str:
        return f"ID: {self.id}，类别: {self.kind}，功能：进入并停留时自动获得工资，每次获得 {self.salary} 元，消耗 {self.relax_cost} relax"

class bed(Interactable):
    """床：可交互物品，供智能体休息恢复 relax。"""

    def __init__(self, id: str, position: list, world, free_num=1):
        super().__init__(id,None, position, world)
        self.kind = "bed"
        self.free_num = free_num
        self.occupant_id = None  # 当前占用者的 agent ID，None 表示无人占用

    def interact(self, agent) -> str:
        if self.free_num <= 0:
            return f"床 {self.id} 已满，无法休息"
        self.occupant_id = agent.id
        self.free_num -= 1
        agent.sleep_status(self.id)
        # 将智能体移到床的坐标（与床重叠，地图上只显示床）
        # 注意：此处已在 operator_tools.sleep() 的 _world_lock 内，不能重复加锁
        old_x, old_y = agent.position
        agent.world.map.remove(old_x, old_y)
        agent.position = list(self.position)
        per_tick = agent.config.sleep_relax_recover / max(1, agent.sleep_ticks_remaining)
        return f"开始在床 {self.id} 上休息，每步恢复 {per_tick:.2f} relax，将休息 {agent.sleep_ticks_remaining} 步"

    def exit_bed(self, agent) -> str:
        if self.occupant_id == agent.id:
            self.occupant_id = None
            self.free_num += 1
            return f"{agent.id} 从床 {self.id} 上醒来，休息结束"
        return f"{agent.id} 不在床 {self.id} 上，无需离开"

    def get_desc(self) -> str:
        return f"ID: {self.id}，类别: {self.kind}，功能：休息恢复，每张床只能同时供 1 个智能体使用，当前空闲床位 {self.free_num}"

class food_shop(building):
    """食品店：建筑子类，智能体进入并停留时自动补充 satiety。"""

    def __init__(self, id: str, position: list, world,food_num=10, provide=2, price=5):
        super().__init__(id, position, world)
        self.kind = "food_shop"
        self.food_num = food_num  # 商品数量
        self.provide = provide  # 提供的饱腹感
        self.price = price  # 价格

    def interact(self, agent) -> str:
        # 食品店必须先检查余额，避免库存减少但智能体没有实际付款。
        if self.food_num <= 0:
            return f"食品店 {self.id} 已售罄"
        if agent.satisfaction.get("money", 0) < self.price:
            return f"余额不足，无法在食品店 {self.id} 购买食物（需要 {self.price} 元）"
        self.food_num -= 1
        agent.update_satisfaction("satiety", self.provide)
        agent.update_satisfaction("money", -self.price)
        return f"在食品店 {self.id} 购买食物，花费 {self.price} 元，补充 {self.provide} satiety"

    def get_desc(self) -> str:
        return f"ID: {self.id}，类别: {self.kind}，功能：进入并停留时自动补充饱腹度，每次花费 {self.price} 元，提供 {self.provide} 饱腹感，剩余商品 {self.food_num}"


class playground(building):
    """游乐场：建筑子类，智能体进入并停留时自动恢复 relax。"""

    def __init__(self, id: str, position: list, world,provide=10,price=3):
        super().__init__(id, position, world)
        self.kind = "playground"
        self.provide = provide  # 提供的放松度
        self.price = price  # 价格

    def interact(self, agent) -> str:
        # 游乐场是付费 relax 恢复渠道，余额不足时不改变任何需求。
        if agent.satisfaction.get("money", 0) < self.price:
            return f"余额不足，无法在游乐场 {self.id} 娱乐（需要 {self.price} 元）"
        agent.update_satisfaction("money", -self.price)
        agent.update_satisfaction("relax", self.provide)
        return f"在游乐场 {self.id} 放松，花费 {self.price} 元，恢复 {self.provide} relax"
    
    def get_desc(self) -> str:
        return f"ID: {self.id}，类别: {self.kind}，功能：放松恢复，每次花费 {self.price} 元，提供 {self.provide} 放松度"
