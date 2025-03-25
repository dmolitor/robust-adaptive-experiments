from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Callable, Dict

class Bandit(ABC):
    """
    An abstract class for Bandit algorithms. Each bandit algorithm must define
    the abstract methods defined below.
    """
    @abstractmethod
    def control(self) -> int:
        """
        Returns the index of the arm that is the control arm. E.g. if the
        bandit is a 3-arm bandit with the first arm being the control arm,
        this should return the value 0.
        """

    @abstractmethod
    def eliminate_arm(self, arm: int) -> None:
        """
        Eliminate an arm of the bandit. In other words, the specified arm
        should no longer ever be assigned as a treatment and all other
        methods should change as necessary.

        Parameters:
        -----------
        arm: int - The index of the eliminated bandit arm
        """
    
    @abstractmethod
    def k(self) -> int:
        """This method that returns the number of arms in the bandit"""
    
    @abstractmethod
    def probabilities(self) -> Dict[int, float]:
        """
        Returns a dictionary with the arm indices as keys and 
        selection probabilities for each arm as values. For example,
        if the bandit algorithm is UCB with three arms, and the third arm has
        the maximum confidence bound, then this should return the following
        dictionary: {0: 0., 1: 0., 1: 1.}, since UCB is deterministic.
        """
    
    @abstractmethod
    def reactivate_arm(self, arm: int):
        """
        This method is a pseudo-inverse of the `eliminate_arm` method. In other
        words this specifies an arm that, at one point, was deactivated, but
        now should be added back to the pool of active treatment arms. It
        should alter all other methods as necessary.
        """

    @abstractmethod
    def reward(self, arm: int) -> "Reward":
        """
        Returns the reward for a selected arm. Returned object must
        be of class `Reward`.
        
        Parameters:
        -----------
        arm: int - The index of the selected bandit arm
        """
    
    @abstractmethod
    def t(self) -> int:
        """
        This method returns the current time step of the bandit, and then
        increments the time step by 1. E.g. if the bandit has completed
        9 iterations, this should return the value 10. Time step starts
        at 1, not 0.
        """

class AB(Bandit):
    """
    A class for implementing and A/B-style experiment
    """
    def __init__(self, k: int, control: int, reward: Callable[[int], float]):
        self._active_arms = [x for x in range(k)]
        self._control = control
        self._k = k
        self._t = 1
        self._reward_fn = reward
    
    def control(self) -> int:
        return self._control
    
    def eliminate_arm(self, arm: int) -> None:
        self._active_arms.remove(arm)
        self._k -= 1
    
    def k(self) -> int:
        return self._k
    
    def probabilities(self) -> Dict[int, float]:
        assert self.k() == len(self._active_arms), "Mismatch in `len(self._active_arms)` and `self.k()`"
        return {x: 1/self.k() for x in self._active_arms}
    
    def reactivate_arm(self, arm: int) -> None:
        self._active_arms.append(arm)
        self._active_arms.sort()
        self._k += 1
    
    def reward(self, arm: int) -> float:
        reward: Reward = self._reward_fn(arm)
        # Assert that the reward is an object of class `Reward`
        if not isinstance(reward, Reward):
            raise ValueError("The provided reward function must return `Reward` objects")
        return reward

    def t(self) -> int:
        step = self._t
        self._t += 1
        return step

class TSNormal(Bandit):
    """
    A class for implementing Thompson Sampling on Normal data
    """
    def __init__(
        self,
        k: int,
        control: int,
        reward: Callable[[int], float],
        optimize: str = "max"
    ):
        self._active_arms = [x for x in range(k)]
        self._control = control
        self._k = k
        self._optimize = optimize
        self._params = {x: {"mean": 0., "var": 10e6} for x in range(k)}
        self._rewards = {x: [] for x in range(k)}
        self._reward_fn = reward
        self._t = 1
    
    def calculate_probs(self) -> Dict[int, float]:
        samples = np.column_stack([
            np.random.normal(
                self._params[idx]["mean"],
                np.sqrt(self._params[idx]["var"]),
                1
            )
            for idx in self._active_arms
        ])
        if self._optimize == "max":
            optimal_indices = np.argmax(samples, axis=1)
        elif self._optimize == "min":
            optimal_indices = np.argmin(samples, axis=1)
        else:
            raise ValueError("`self._optimal` must be one of: ['max', 'min']")
        win_counts = {
            idx: np.sum(optimal_indices == i) / 1
            for i, idx in enumerate(self._active_arms)
        }
        return win_counts

    def control(self) -> int:
        return self._control
    
    def eliminate_arm(self, arm: int) -> None:
        self._active_arms.remove(arm)
        self._k -= 1
    
    def k(self) -> int:
        return self._k
    
    def probabilities(self) -> Dict[int, float]:
        assert self.k() == len(self._active_arms), "Mismatch in `len(self._active_arms)` and `self.k()`"
        probs = self.calculate_probs()
        return probs
    
    def reactivate_arm(self, arm: int) -> None:
        self._active_arms.append(arm)
        self._active_arms.sort()
        self._k += 1

    def reward(self, arm: int) -> float:
        reward: Reward = self._reward_fn(arm)
        # Assert that the reward is an object of class `Reward`
        if not isinstance(reward, Reward):
            raise ValueError("The provided reward function must return `Reward` objects")
        self._rewards[arm].append(reward.outcome)
        var = (5/4)**2
        var_prior = self._params[arm]["var"]
        var_posterior = 1/(1/var_prior + 1/var)
        mean_prior = self._params[arm]["mean"]
        mean_posterior = var_posterior*(mean_prior/var_prior + reward.outcome/var)
        self._params[arm]["mean"] = mean_posterior
        self._params[arm]["var"] = var_posterior
        return reward

    def t(self) -> int:
        step = self._t
        self._t += 1
        return step

class TSBernoulli(Bandit):
    """
    A class for implementing Thompson Sampling on Bernoulli data
    """
    def __init__(
        self,
        k: int,
        control: int,
        reward: Callable[[int], float],
        optimize: str = "max"
    ):
        self._active_arms = [x for x in range(k)]
        self._control = control
        self._k = k
        self._means = {x: 0. for x in range(k)}
        self._optimize = optimize
        self._params = {x: {"alpha": 1, "beta": 1} for x in range(k)}
        self._rewards = {x: [] for x in range(k)}
        self._reward_fn = reward
        self._t = 1
    
    def calculate_probs(self) -> Dict[int, float]:
        sample_size = 1
        samples = np.column_stack([
            np.random.beta(
                a=self._params[idx]["alpha"],
                b=self._params[idx]["beta"],
                size=sample_size
            )
            for idx in self._active_arms
        ])
        if self._optimize == "max":
            optimal_indices = np.argmax(samples, axis=1)
        elif self._optimize == "min":
            optimal_indices = np.argmin(samples, axis=1)
        else:
            raise ValueError("`self._optimal` must be one of: ['max', 'min']")
        win_counts = {
            idx: np.sum(optimal_indices == i) / sample_size
            for i, idx in enumerate(self._active_arms)
        }
        return win_counts

    def control(self) -> int:
        return self._control
    
    def eliminate_arm(self, arm: int) -> None:
        self._active_arms.remove(arm)
        self._k -= 1
    
    def k(self) -> int:
        return self._k
    
    def probabilities(self) -> Dict[int, float]:
        assert self.k() == len(self._active_arms), "Mismatch in `len(self._active_arms)` and `self.k()`"
        probs = self.calculate_probs()
        return probs
    
    def reactivate_arm(self, arm: int) -> None:
        self._active_arms.append(arm)
        self._active_arms.sort()
        self._k += 1
    
    def reward(self, arm: int) -> float:
        reward: Reward = self._reward_fn(arm)
        # Assert that the reward is an object of class `Reward`
        if not isinstance(reward, Reward):
            raise ValueError("The provided reward function must return `Reward` objects")
        self._rewards[arm].append(reward.outcome)
        if reward.outcome == 1:
            self._params[arm]["alpha"] += 1
        else:
            self._params[arm]["beta"] += 1
        self._means[arm] = (
            self._params[arm]["alpha"]
            /(self._params[arm]["alpha"] + self._params[arm]["beta"])
        )
        return reward

    def t(self) -> int:
        step = self._t
        self._t += 1
        return step

class UCB(Bandit):
    """
    A class for implementing the UCB algorithm.
    """
    def __init__(
        self,
        k: int,
        control: int,
        reward: Callable[[int], float],
        optimize: str = "max"
    ):
        self._active_arms = [x for x in range(k)]
        self._control = control
        self._k = k
        self._optimize = optimize  # "max" for UCB1, "min" for lower confidence bound
        self._reward_fn = reward
        self._t = 1
        # Track counts, cumulative rewards, means, and reward history for each arm.
        self._counts = {x: 0 for x in range(k)}
        self._sums = {x: 0.0 for x in range(k)}
        self._means = {x: 0.0 for x in range(k)}
        self._rewards = {x: [] for x in range(k)}
    
    def calculate_probs(self) -> Dict[int, float]:
        # If any arm hasn't been pulled yet, assign equal probability among them.
        zero_count_arms = [arm for arm in self._active_arms if self._counts[arm] == 0]
        if zero_count_arms:
            prob = 1.0 / len(zero_count_arms)
            probs = {
                arm: prob if arm in zero_count_arms else 0.0
                for arm in self._active_arms
            }
            return probs
        # Compute UCB (or lower confidence bound for "min" optimization).
        ucb_values = {}
        for arm in self._active_arms:
            avg = self._sums[arm] / self._counts[arm]
            bonus = np.sqrt((2 * np.log(self._t)) / self._counts[arm])
            if self._optimize == "max":
                ucb_values[arm] = avg + bonus
            elif self._optimize == "min":
                ucb_values[arm] = avg - bonus
            else:
                raise ValueError("`optimize` must be one of: ['max', 'min']")
        if self._optimize == "max":
            best_arm = max(ucb_values, key=ucb_values.get)
        elif self._optimize == "min":
            best_arm = min(ucb_values, key=ucb_values.get)
        probs = {
            arm: 1.0 if arm == best_arm else 0.0
            for arm in self._active_arms
        }
        return probs
    
    def control(self) -> int:
        return self._control
    
    def eliminate_arm(self, arm: int) -> None:
        self._active_arms.remove(arm)
        self._k -= 1
    
    def k(self) -> int:
        return self._k
    
    def probabilities(self) -> Dict[int, float]:
        assert self.k() == len(self._active_arms), "Mismatch in active arms and k"
        probs = self.calculate_probs()
        return probs
    
    def reactivate_arm(self, arm: int) -> None:
        if arm not in self._active_arms:
            self._active_arms.append(arm)
            self._active_arms.sort()
            self._k += 1
    
    def reward(self, arm: int) -> float:
        reward: Reward = self._reward_fn(arm)
        # Assert that the reward is an object of class `Reward`
        if not isinstance(reward, Reward):
            raise ValueError("The provided reward function must return `Reward` objects")
        self._rewards[arm].append(reward.outcome)
        self._counts[arm] += 1
        self._sums[arm] += reward.outcome
        self._means[arm] = self._sums[arm] / self._counts[arm]
        return reward
    
    def t(self) -> int:
        step = self._t
        self._t += 1
        return step

class Reward:
    """
    A simple class for reward functions. It has two attributes: `self.outcome`
    and `self.covariates`. Each reward function should return a reward object.
    For covariate adjusted algorithms, the reward should contain both the
    outcome as well as the corresponding covariates. For non-covariate adjusted
    algorithms, only the outcome should be specified.
    """
    def __init__(self, outcome: float, covariates: pd.DataFrame | None = None):
        self.outcome = outcome
        self.covariates = covariates