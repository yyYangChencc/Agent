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

    def interact(self, agent) -> str:
        # 占位实现：等待 work 动作系统接入后再扩展
        return f"与建筑 {self.id} 互动（暂未实现具体效果）"

    def get_desc(self) -> str:
        return f"ID: {self.id}，类别: {self.kind}"


class bed(building):
    """床：建筑子类，供智能体休息恢复 relax。具体交互效果待 sleep 动作接入后实现。"""

    def __init__(self, id: str, position: list, world):
        super().__init__(id, position, world)
        self.kind = "bed"
        self.free_num = 1  # 床位数量

    def interact(self, agent) -> str:
        # 占位实现：等待 sleep 动作系统接入后再扩展
        return f"在床 {self.id} 上休息（暂未实现具体效果）"

    def get_desc(self) -> str:
        return f"ID: {self.id}，类别: {self.kind}，功能：休息恢复，剩余床位 {self.free_num}"

class food_shop(building):
    """食品店：建筑子类，供智能体购买食物补充 satiety。具体交互效果待购买动作接入后实现。"""

    def __init__(self, id: str, position: list, world,food_num=10, provide=2, price=5):
        super().__init__(id, position, world)
        self.kind = "food_shop"
        self.food_num = food_num  # 商品数量
        self.provide = provide  # 提供的饱腹感
        self.price = price  # 价格

    def interact(self, agent) -> str:
        # 占位实现：等待购买动作系统接入后再扩展
        return f"在食品店 {self.id} 购买食物（暂未实现具体效果）"

    def get_desc(self) -> str:
        return f"ID: {self.id}，类别: {self.kind}，功能：购买食物，每次购买花费 {self.price} 元，提供 {self.provide} 饱腹感，剩余商品 {self.food_num}"


class playground(building):
    """游乐场：建筑子类，供智能体放松恢复 relax。具体交互效果待娱乐动作接入后实现。"""

    def __init__(self, id: str, position: list, world,provide=0.2,price=3):
        super().__init__(id, position, world)
        self.kind = "playground"
        self.provide = provide  # 提供的放松度
        self.price = price  # 价格

    def interact(self, agent) -> str:
        # 占位实现：等待娱乐动作系统接入后再扩展
        return f"在游乐场 {self.id} 放松（暂未实现具体效果）"
    
    def get_desc(self) -> str:
        return f"ID: {self.id}，类别: {self.kind}，功能：放松恢复，每次花费 {self.price} 元，提供 {self.provide} 放松度"