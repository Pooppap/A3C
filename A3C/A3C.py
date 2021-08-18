import os
import time
import torch

from gym import spaces
from copy import deepcopy
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import explained_variance 


class A3C(A2C):
    def __init__(
        self,
        policy,
        env,
        *args,
        **kwargs
    ):
        shared_policy_set = False
        if "shared_policy" in kwargs:
            kwargs["policy"] = deepcopy(kwargs["shared_policy"])
            self.shared_policy = kwargs.pop("shared_policy")
            # self.shared_policy.optimizer = torch.optim.Adam(self.shared_policy.parameters(), lr=1e-5) # shift optimizer to child memory space

            shared_policy_set = True

        if "base_log_dir" in kwargs:
            self.base_log_dir = kwargs.pop("base_log_dir")
        else:
            self.base_log_dir = None

        self._parent_args = args
        self._parent_kwargs = kwargs

        super().__init__(policy, env, *args, **kwargs)
        if not shared_policy_set:
            self.shared_policy = deepcopy(self.policy)
            # del self.shared_policy.optimizer # No need for global optimizer
            self.shared_policy.share_memory()
        # else:
        #     del self.policy.optimizer

    def ensure_shared_grads(self):
        for param, shared_param in zip(self.policy.parameters(), self.shared_policy.parameters()):
            if shared_param.grad is not None:
                return

            # Correctly cast param to the correct device.
            shared_param._grad = param.grad.to(shared_param.device)

    def _multiprocess_learning(self, rank, *args):
        host_torch_seed = self.seed if self.seed else torch.seed
        host_torch_seed += rank

        env = deepcopy(self.env)
        env = Monitor(env, filename=os.path.join(self.base_log_dir, f"Process_{rank}"))

        model = A3C(deepcopy(self.policy), env, *self._parent_args, shared_policy=self.shared_policy, **self._parent_kwargs)
        model._learn(*args, **self._learn_kwargs)

    def learn(self, *args, **kwargs):
        nprocs = kwargs.pop("nprocs", 1)
        lock = torch.multiprocessing.Lock()
        counter = torch.multiprocessing.Value('i', 0)

        kwargs["is_child"] = True
        self._learn_kwargs = kwargs
        args = [lock, counter] + args
        torch.multiprocessing.spawn(self._multiprocess_learning, args=args, nprocs=nprocs)


    def _train(self):
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.train()

        # Update optimizer learning rate
        self._update_learning_rate(self.shared_policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            # TODO: avoid second computation of everything because of the gradient
            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = torch.nn.functional.mse_loss(rollout_data.returns, values)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -torch.mean(-log_prob)
            else:
                entropy_loss = -torch.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # Optimization step
            self.shared_policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.ensure_shared_grads()
            self.shared_policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

    def _learn(
        self,
        lock,
        counter,
        total_timesteps,
        callback=None,
        log_interval=1,
        eval_env=None,
        eval_freq=-1,
        n_eval_episodes=5,
        tb_log_name="OnPolicyAlgorithm",
        eval_log_path=None,
        reset_num_timesteps=True,
    ):
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            self.policy.load_state_dict(self.shared_policy.state_dict())

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
            
            with lock:
                counter.value = self.num_timesteps

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self):
        state_dicts = ["shared_policy", "shared_policy.optimizer"]

        return state_dicts, []
