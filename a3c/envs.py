import gym
import logging
import numpy as np
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_ENV_ID = 'Breakout-v0'
PS_IP = '127.0.0.1'
PS_PORT = 12222
TB_PORT = 12345
NUM_GLOBAL_STEPS = 300000
SAVE_MODEL_SECS = 30
SAVE_SUMMARIES_SECS = 30
LOG_DIR = './log/'
NUM_WORKER = 4
VISUALISED_WORKERS = []  # e.g. [0] or [1,2]

_N_AVERAGE = 100

VSTR = 'V0'


class MyLastN:
    def __init__(self, n):
        self._arr = []
        self._n = n

    def append(self, o):
        self._arr.append(o)
        if len(self._arr) > self._n:
            self._arr.pop(0)

    def average(self):
        return np.average(self._arr) if len(self._arr) > 0 else 0

    def last(self):
        return self._arr[-1] if len(self._arr) > 0 else 0


class MyEnv:
    def __init__(self, env):
        self._env = env
        self._ep_count = 0
        self._ep_steps = 0
        self._ep_rewards = []

        self._steps_last_n_eps = MyLastN(_N_AVERAGE)
        self._rewards_last_n_eps = MyLastN(_N_AVERAGE)

        self.spec = env.spec
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = env.metadata

    def reset(self):
        self._ep_count += 1
        self._ep_steps = 0
        self._ep_rewards = []
        s = self._env.reset()
        return s

    def step(self, a):
        s, r, t, _ = self._env.step(a)

        self._ep_steps += 1
        self._ep_rewards.append(r)

        i = {}
        if t:
            self._steps_last_n_eps.append(self._ep_steps)
            self._rewards_last_n_eps.append(np.sum(self._ep_rewards))

            i['{}/ep_count'.format(VSTR)] = self._ep_count
            i['{}/ep_steps'.format(VSTR)] = self._steps_last_n_eps.last()
            i['{}/ep_rewards'.format(VSTR)] = self._rewards_last_n_eps.last()
            i['{}/aver_steps_{}'.format(VSTR, _N_AVERAGE)] = self._steps_last_n_eps.average()
            i['{}/aver_rewards_{}'.format(VSTR, _N_AVERAGE)] = self._rewards_last_n_eps.average()

            print(i)

        return s, r, t, i


def create_env(task_id):
    env = gym.make(_ENV_ID)
    return MyEnv(env)


