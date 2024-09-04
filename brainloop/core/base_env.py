# base_env.py

class BaseEnv:

  def __init__(self):
    raise NotImplementedError

  def step(self, action):
    raise NotImplementedError

  def reset(self):
    raise NotImplementedError


  def render(self):
    raise NotImplementedError


  def close(self):
    raise NotImplementedError
