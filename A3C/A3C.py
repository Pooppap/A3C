import gym

from copy import deepcopy
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env


class A3C(A2C):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, kwargs)
        self.shared_policy = deepcopy(self.policy)
        self.shared_policy.share_memory()

    def ensure_shared_grads(self):
        for param, shared_param in zip(self.policy.parameters(), self.shared_policy.parameters()):
            if shared_param.grad is not None:
                return

            # Correctly cast param to the correct device.
            shared_param._grad = param.grad.to(shared_param.device)


