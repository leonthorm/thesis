import gymnasium as gym
import dynobench

class DynoColtransEnv(gym.Env):
    def __init__(self):
        super().__init__()

    def step(self, action):
        # Do something, depending on the action and current value of mu the next state is computed
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info