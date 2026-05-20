from tools.base import Tool
# 死代码：get_agent_memory 引用不存在的 agent.mem.agent_profile，此文件未被任何模块调用
def register_mem_tools(mem):
    tool_specs = {
        "get_agent_memory": {
            "description": "获得指定ID的人物记忆，若有多个人物，则用空格分开",
            "args": {"ID": str},
            "returns":str 
        }
    }

    tools = {}
    for name, spec in tool_specs.items():
        func = getattr(mem, name, None)
        if func is None:
            raise ValueError(f"operator 未实现方法: {name}")

        tools[name] = Tool(
            name=name,
            func=func,
            description=spec["description"],
            args=spec["args"]
        )
    lines = []
    lines.append("你是一个智能体，可以使用以下记忆工具进行回忆以辅助思考：\n")

    for i, (name, spec) in enumerate(tool_specs.items(), start=1):
        lines.append(f"{i}. 工具名：{name}")
        lines.append(f"   功能：{spec['description']}")

        # 参数
        lines.append("   参数：")
        for arg, arg_type in spec["args"].items():
            type_name = arg_type.__name__
            lines.append(f"     - {arg} ({type_name})")

        # 返回值
        return_type = spec["returns"].__name__
        lines.append(f"   返回：{return_type}")

        # 约束
        if "constraint" in spec:
            lines.append(f"   约束：{spec['constraint']}")

        lines.append("")  # 空行分隔

    return tools,'\n'.join(lines)


class MemOperator:
    '''
    MemOperator 的 Docstring
    智能体记忆相关操作，包括记忆的存储、提取与修改
    '''
    def __init__(self,world):
        self.world=world
        self.mem_tools, self.mem_tools_prompt = register_mem_tools(self)

    def get_agent_memory(self,operator_ID,ID:str):
        ID = ID.split()
        agent = self.world.agents[operator_ID]
        res = []
        for i in ID:
            t = []
            if agent.mem.agent_profile.get(i,"None") != "None":
                for key,value in agent.mem.agent_profile[i].items():
                    t.append(f"{key}:{value}")
                res.append(f"{i}人物档案   {' '.join(t)}")
            else:
                res.append(f"无{i}相关记忆")
        return "\n".join(res)
    