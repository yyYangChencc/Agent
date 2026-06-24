from __future__ import annotations

import os
import sys
import types
import unittest


BACKEND_DIR = os.path.dirname(__file__)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


if "chromadb" not in sys.modules:
    chromadb_stub = types.ModuleType("chromadb")

    class _PersistentClient:
        def __init__(self, *args, **kwargs):
            pass

    chromadb_stub.PersistentClient = _PersistentClient
    sys.modules["chromadb"] = chromadb_stub

if "chromadb.config" not in sys.modules:
    chromadb_config_stub = types.ModuleType("chromadb.config")

    class Settings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    chromadb_config_stub.Settings = Settings
    sys.modules["chromadb.config"] = chromadb_config_stub


from persona.config import AgentConfig
from persona.runtime import SimulationRuntime


class _LLMStub:
    pass


class _MemoryStub:
    def __init__(self):
        self.reset_count = 0

    def reset_all(self):
        self.reset_count += 1


class RuntimeResetTest(unittest.TestCase):
    def test_reset_preserves_llm_backed_opinion_assessor(self):
        runtime = SimulationRuntime(
            config=AgentConfig(opinion_assessment_mode="llm"),
            llm=_LLMStub(),
            mem=_MemoryStub(),
            world=None,
            platform=None,
            policy=None,
            social_policy=None,
            conv_policy=None,
            reflect=None,
        )

        runtime.reset()

        self.assertIs(runtime.world.opinion_assessor.llm, runtime.llm)
        self.assertIs(runtime.world.psychological_assessor.llm, runtime.llm)
        self.assertEqual(runtime.mem.reset_count, 1)


if __name__ == "__main__":
    unittest.main()
