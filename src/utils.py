import numpy as np
from typing import Any, List

def ate(ite: List[float]) -> float:
    """Unbiased ATE estimator. Sample mean of unbiased ITE estimates"""
    return np.mean(ite).astype(float)

def check_shrinkage_rate(t: int, delta_t: float):
    """Checks the shrinkage rate delta_t defined in Liang and Bojinov"""
    assert t <= 1 or delta_t > 1/(t**(1/4)), "Sequence is converging to 0 too quickly"

def cs_radius(var: List[float], t: int, t_star: int, alpha: float = 0.05) -> float:
    """
    Confidence sequence radius
    
    Parameters:
    -----------
    var   : An array-like of individual treatment effect variances (upper bounds)
    t     : The current time-step of the algorithm
    t_star: The time-step at which we want to optimize the CSs to be tightest
    alpha : The size of the statistical test

    Return:
    -------
    The radius of the Confidence Sequence. Aka the value V such that
    tau (ATE estimate) ± V is a valid alpha-level CS.
    """
    S = np.sum(var)
    eta = np.sqrt((-2*np.log(alpha) + np.log(-2*np.log(alpha) + 1))/t_star)
    rad = np.sqrt(
        2*(S*(eta**2) + 1)/((t**2)*(eta**2))
        * np.log(np.sqrt(S*(eta**2) + 1)/alpha)
    )
    return rad.astype(float)

def ite(outcome: float, treatment: int, propensity: float) -> float:
    """Unbiased individual treatment effect estimator"""
    if treatment == 0:
        ite = -outcome/propensity
    else:
        ite = outcome/propensity
    return ite

def last(x: List[Any]) -> Any:
    return x[len(x) - 1]

def var(outcome: float, propensity: float) -> float:
    """Upper bound for individual treatment effect variance"""
    var_ub = (outcome**2)/(propensity**2)
    return var_ub

def weighted_probs(
    bandit_probs: np.ndarray | list[float], 
    weights: np.ndarray | list[float]
) -> np.ndarray:
    """
    Update bandit probabilities based on weights, redistributing lost mass.
    
    Parameters:
    - bandit_probs (Union[np.ndarray, list[float]]): Original probabilities (must sum to 1).
    - weights (Union[np.ndarray, list[float]]): Importance weights (values in [0, 1]).
    
    Returns:
    - np.ndarray: Updated bandit probabilities.
    """
    bandit_probs = np.asarray(bandit_probs)
    weights = np.asarray(weights)
    assert np.isclose(np.sum(bandit_probs), 1., rtol=0.01), "Bandit probabilities should sum to 1"
    updated_probs = weights * bandit_probs
    losses = bandit_probs * (1 - weights)
    total_loss = np.sum(losses)
    relative_weights = weights / np.sum(weights)
    redistributed_loss = total_loss * relative_weights
    updated_bandit_probs = updated_probs + redistributed_loss
    assert np.isclose(np.sum(updated_bandit_probs), 1., rtol=0.01), "Updated bandit probabilities should sum to 1"
    return updated_bandit_probs
