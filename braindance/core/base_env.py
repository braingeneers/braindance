# base_env.py
import time


class BaseEnv:

    def __init__(self, max_time_sec=60, verbose=1):
        self.max_time_sec = max_time_sec
        self.verbose = verbose

    def _init_time_management(self):
        self.start_time = self.cur_time = time.perf_counter()

    def time_elapsed(self):
        '''Returns time since initialization of the environment.'''
        return time.perf_counter() - self.start_time

    @property
    def dt(self):
        '''Returns time since the last step.'''
        return time.perf_counter() - self.cur_time

    def _check_if_done(self):
        if self.time_elapsed() > self.max_time_sec:
            # Debugging
            if self.verbose >= 1:
                print(
                    f'Max time {self.max_time_sec} reached at {self.time_elapsed():.1f}')
            self._cleanup()

            return True
        return False

    def _cleanup(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
