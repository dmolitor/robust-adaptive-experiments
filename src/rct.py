import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm
from typing import Callable, List

generator = np.random.default_rng(seed=123)

def rct(
    t: int,
    k: int,
    control: int,
    reward_fn: Callable[[int], float],
    alpha: float = 0.05
):
    outcomes = []
    def ci(treat: List[float], control: List[float], alpha: float = 0.05):
        mean = np.mean(treat)
        mean_control = np.mean(control)
        stderr = np.std(treat, ddof=1)/np.sqrt(len(treat))
        stderr_control = np.std(control, ddof=1)/np.sqrt(len(control))
        mean_diff = mean - mean_control
        stderr_diff = np.sqrt(stderr**2 + stderr_control**2)
        z_score = stats.norm.ppf(1 - alpha/2)
        lower_bound = mean_diff - z_score * stderr_diff
        upper_bound = mean_diff + z_score * stderr_diff
        return(mean_diff, lower_bound, upper_bound)

    for _ in range(k):
        outcomes.append([])
    for i in tqdm(range(1, t + 1), total=t):
        arm = generator.multinomial(1, pvals=[1/k]*k).argmax()
        outcome = reward_fn(arm)
        outcomes[arm].append(outcome)
    rct_results = {"arm": [], "ate": [], "lb": [], "ub": [], "n": []}
    for arm in range(k):
        if arm == control:
            continue
        rct_results["arm"].append(arm)
        ate, lb, ub = ci(outcomes[arm], outcomes[control], alpha=alpha)
        rct_results["ate"].append(ate)
        rct_results["lb"].append(lb)
        rct_results["ub"].append(ub)
        rct_results["n"].append(len(outcomes[arm]))
    return pd.DataFrame(rct_results)