import logging
import sys
from pathlib import Path
from datetime import datetime


LOG_FORMAT = "%(asctime)s | tick=%(tick)s | %(levelname)-8s | %(name)s | %(message)s"
_current_simulation_tick = 0


def set_log_tick(tick: int) -> None:
    """更新后续全部日志记录使用的仿真时间步。"""

    global _current_simulation_tick
    _current_simulation_tick = max(0, int(tick))


def get_log_tick() -> int:
    """返回当前日志时间步。"""

    return _current_simulation_tick


class SimulationTickFilter(logging.Filter):
    """为每条控制台和文件日志补充统一的 tick 字段。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.tick = get_log_tick()
        return True


def setup_logging(level: str = "INFO", log_dir: str = "logs") -> None:
    """Configure root logger with console (INFO) and file (DEBUG) handlers."""
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"agent_{timestamp}.log"

    formatter = logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S")
    tick_filter = SimulationTickFilter()

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console: only INFO and above
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(formatter)
    console.addFilter(tick_filter)

    # File: full DEBUG detail
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(tick_filter)

    root.addHandler(console)
    root.addHandler(file_handler)

    # Suppress verbose HTTP request logs from the OpenAI client's transport layer
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)

    logging.getLogger(__name__).info("日志系统已启动，日志文件: %s", log_file)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
