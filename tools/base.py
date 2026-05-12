class Tool:
    def __init__(self, name, func, description, args):
        self.name = name
        self.func = func
        self.description = description
        self.args = args

    def run(self, **kwargs):
        return self.func(**kwargs)
