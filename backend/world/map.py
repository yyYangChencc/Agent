from persona.logger import get_logger

logger = get_logger(__name__)

class Map:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [['0' for _ in range(width)] for _ in range(height)]

    def is_empty(self, x, y):
        return self.grid[x][y] == '0'

    def place(self, x, y, obj_id):
        self.grid[x][y] = obj_id

    def remove(self, x, y):
        self.grid[x][y] = '0'

    def get_e(self, x, y):
        return self.grid[x][y]

    def print_map(self):
        rows = []
        for i in range(self.width):
            rows.append(' '.join(self.grid[i][j] for j in range(self.height)))
        logger.debug("地图状态:\n%s", '\n'.join(rows))
