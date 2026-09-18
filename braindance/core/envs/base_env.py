#!/usr/bin/env python3
"""
base_env.py
A minimal base environment providing timing and abstract methods.
"""

import time
from abc import ABC, abstractmethod

class BaseEnv(ABC):
    def __init__(self, max_time_sec: float = 60, verbose: int = 1):
        self.max_time_sec = max_time_sec
        self.verbose = verbose
        self._init_time_management()

    def _init_time_management(self):
        self.start_time = self.cur_time = time.perf_counter()

    def time_elapsed(self) -> float:
        return time.perf_counter() - self.start_time

    @property
    def dt(self) -> float:
        return time.perf_counter() - self.cur_time

    def _check_if_done(self) -> bool:
        if self.time_elapsed() > self.max_time_sec:
            if self.verbose:
                print(f'Max time {self.max_time_sec} sec reached at {self.time_elapsed():.1f} sec')
            self._cleanup()
            return True
        return False

    @abstractmethod
    def step(self, action=None):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def close(self):
        pass
