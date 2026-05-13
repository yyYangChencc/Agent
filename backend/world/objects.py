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

    def interact(self, agent) -> str:
        agent.update_need("satiety", self.provide)
        self.eaten()
        return f"吃到了{self.id}"

    def eaten(self):
        self.num -= 1
        if self.num <= 0:
            self.world.map.remove(*self.position)
            self.world.objects.pop(self.id)
