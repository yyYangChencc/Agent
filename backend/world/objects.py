class objects:
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
    """Base class for objects that can be interacted with."""

    def interact(self, agent) -> str:
        raise NotImplementedError

class Uninteractable(objects):
    """Base class for objects that can not be interacted with."""


class food(Interactable):
    def __init__(self, id: str, num: int, provide: int, position: list, world):
        super().__init__(id, num, position, world)
        self.provide = provide
        self.kind = "food"

    def get_desc(self) -> str:
        return f"ID: {self.id}，类别：{self.kind}，功能：提供 {self.provide} 饱腹感，剩余数量 {self.num}"
        
    def interact(self, agent) -> str:
        agent.update_need("satiety", self.provide)
        self.eaten()
        return f"吃到了{self.id}"

    def eaten(self):
        self.num -= 1
        if self.num <= 0:
            self.world.map.remove(*self.position)
            self.world.objects.pop(self.id)


class building(Interactable):
    """场景中的建筑，作为可交互物品占据地图格位。具体交互动作（如 work）后续接入。"""

    def __init__(self, id: str, position: list, world):
        # 建筑无数量属性，num 固定为 None（前端按"无限"渲染，不会因 num<=0 而隐藏）
        super().__init__(id, None, position, world)
        self.kind = "building"
        self.occupants: list[str] = []

    def interact(self, agent) -> str:
        # 占位实现：等待 work 动作系统接入后再扩展
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
    """公司：建筑子类，供智能体工作赚钱。具体交互效果待 work 动作接入后实现。"""

    def __init__(self, id: str, position: list, world, salary=10):
        super().__init__(id, position, world)
        self.kind = "company"
        self.salary = salary  # 工作获得的工资

    def interact(self, agent) -> str:
        agent.update_need("money", self.salary)
        agent.update_need("relax", -10)
        return f"在公司 {self.id} 工作，获得 {self.salary} 元工资，消耗 10 relax"
    
    def get_desc(self) -> str:
        return f"ID: {self.id}，类别: {self.kind}，功能：工作赚钱，每次工作获得 {self.salary} 元工资"

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
        return f"开始在床 {self.id} 上休息，已恢复 {agent.config.sleep_relax_recover} relax，将休息 {agent.sleep_ticks_remaining} 步"

    def exit_bed(self, agent) -> str:
        if self.occupant_id == agent.id:
            self.occupant_id = None
            self.free_num += 1
            return f"{agent.id} 从床 {self.id} 上醒来，休息结束"
        return f"{agent.id} 不在床 {self.id} 上，无需离开"

    def get_desc(self) -> str:
        return f"ID: {self.id}，类别: {self.kind}，功能：休息恢复，每张床只能同时供 1 个智能体使用，当前空闲床位 {self.free_num}"

class food_shop(building):
    """食品店：建筑子类，供智能体购买食物补充 satiety。具体交互效果待购买动作接入后实现。"""

    def __init__(self, id: str, position: list, world,food_num=10, provide=2, price=5):
        super().__init__(id, position, world)
        self.kind = "food_shop"
        self.food_num = food_num  # 商品数量
        self.provide = provide  # 提供的饱腹感
        self.price = price  # 价格

    def interact(self, agent) -> str:
        if self.food_num <= 0:
            return f"食品店 {self.id} 已售罄"
        self.food_num -= 1
        agent.update_need("satiety", self.provide)
        agent.update_need("money", -self.price)
        return f"在食品店 {self.id}"

    def get_desc(self) -> str:
        return f"ID: {self.id}，类别: {self.kind}，功能：购买食物，每次购买花费 {self.price} 元，提供 {self.provide} 饱腹感，剩余商品 {self.food_num}"


class playground(building):
    """游乐场：建筑子类，供智能体放松恢复 relax。具体交互效果待娱乐动作接入后实现。"""

    def __init__(self, id: str, position: list, world,provide=10,price=3):
        super().__init__(id, position, world)
        self.kind = "playground"
        self.provide = provide  # 提供的放松度
        self.price = price  # 价格

    def interact(self, agent) -> str:
        if agent.need.get("money", 0) < self.price:
            return f"余额不足，无法在游乐场 {self.id} 娱乐（需要 {self.price} 元）"
        agent.update_need("money", -self.price)
        agent.update_need("relax", self.provide)
        return f"在游乐场 {self.id} 放松，花费 {self.price} 元，恢复 {self.provide} relax"
    
    def get_desc(self) -> str:
        return f"ID: {self.id}，类别: {self.kind}，功能：放松恢复，每次花费 {self.price} 元，提供 {self.provide} 放松度"