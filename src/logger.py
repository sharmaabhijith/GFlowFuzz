import os
import threading
from enum import IntEnum
from datetime import datetime

_GLOBAL_LOG_DIR = None

def set_global_log_dir(log_dir):
    """Set the global log directory for all loggers."""
    global _GLOBAL_LOG_DIR
    _GLOBAL_LOG_DIR = log_dir
    os.makedirs(log_dir, exist_ok=True)

def get_global_log_dir():
    """Get the global log directory."""
    return _GLOBAL_LOG_DIR

class LEVEL(IntEnum):
    INFO = 1
    TRACE = 2
    VERBOSE = 3

class GlobberLogger:
    """
    GlobberLogger is the recommended logger for all GFlowFuzz components.
    It supports log levels, thread-safe file and console output, and timestamped messages.
    All log files are written to the global log directory set by set_global_log_dir().
    Usage:
        from GFlowFuzz.logger import set_global_log_dir, GlobberLogger, LEVEL
        set_global_log_dir('logs/exp_001')
        logger = GlobberLogger('fuzzer.log', level=LEVEL.TRACE)
        logger.log('message', LEVEL.INFO)
    """
    def __init__(self, file_name: str, level: LEVEL = LEVEL.INFO, console: bool = True):
        log_dir = get_global_log_dir()
        if log_dir is None:
            raise RuntimeError("Global log directory not set. Call set_global_log_dir() first.")
        self.logfile = os.path.join(log_dir, file_name)
        self.level = level
        self.console = console
        self._lock = threading.Lock()
        os.makedirs(log_dir, exist_ok=True)

    def _format_log(self, msg, level: LEVEL):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"[{timestamp}][{level.name}] {msg}"

    def log(self, msg, level: LEVEL = LEVEL.INFO):
        if level > self.level:
            return
        formatted = self._format_log(msg, level)
        with self._lock:
            try:
                with open(self.logfile, "a+", encoding="utf-8") as logfile:
                    logfile.write(formatted + "\n")
                if self.console:
                    print(formatted)
            except Exception as e:
                # Optionally, print to stderr or ignore
                pass

# Usage example (in any component):
# from GFlowFuzz.logger import GlobberLogger, LEVEL
# logger = GlobberLogger("./logs", "component.log", level=LEVEL.TRACE)
# logger.log("This is a trace message", LEVEL.TRACE) 