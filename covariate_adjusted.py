from pathlib import Path
base_dir = Path(__file__).resolve().parent

import joblib
import numpy as np
import pandas as pd
import plotnine as pn
from scipy.stats import t
from typing import Dict, Tuple

from src.bandit import Reward, TSBernoulli, TSNormal, UCB
from src.mad import MAD, MADModified, MADCovariateAdjusted
from src.model import LassoModel, LogitModel, OLSModel
from src.utils import last

generator = np.random.default_rng(seed=123)

# Binary treatment with Normally distributed rewards --------------------------

def reward_covar_adj(arm: int) -> Reward:
    ate = {0: 0.0, 1: 1.3}
    # Draw X values randomly (here using standard normal distribution)
    X1 = np.random.randn()
    X2 = np.random.randn()
    X3 = np.random.randn()
    # Get the corresponding ATE from the dictionary
    ate = ate[arm]
    # Compute Y_i using the given model
    mean = 0.5 + ate + 0.3 * X1 + 1.0 * X2 - 0.5 * X3
    Y_i = generator.normal(mean, 1)
    X_df = pd.DataFrame({"X_1": [X1], "X_2": [X2], "X_3": [X3]})
    reward = Reward(outcome=float(Y_i), covariates=X_df)
    return reward

def reward_vanilla(arm: int) -> Reward:
    reward = reward_covar_adj(arm=arm)
    reward = Reward(outcome=reward.outcome)
    return reward

# Vanilla MAD algorithm for 5000 iterations
mad = MAD(
    # bandit=TSNormal(k=2, control=0, reward=reward_vanilla),
    bandit=UCB(k=2, control=0, reward=reward_vanilla),
    alpha=0.05,
    delta=lambda x: 1./(x**0.24),
    t_star=int(5e3)
)
mad.fit(cs_precision=0.1, verbose=True, early_stopping=False, mc_adjust=None)
(
    mad.plot_ate_path()
    + pn.coord_cartesian(ylim=(-1, 3))
    + pn.geom_hline(yintercept=1.3, linetype="dashed")
    + pn.theme(strip_text_x=pn.element_blank())
).save(
    base_dir / "figures" / "mad_no_covar_ate_path.png",
    width=4,
    height=3,
    dpi=500
)

# Covariate adjusted MAD algorithm for 2000 iterations
mad_covar_adj = MAD(
    # bandit=TSNormal(k=2, control=0, reward=reward_covar_adj),
    bandit=UCB(k=2, control=0, reward=reward_covar_adj),
    alpha=0.05,
    delta=lambda x: 1./(x**0.24),
    t_star=int(5e3),
    model=OLSModel,
    pooled=False,
    n_warmup=50
)
mad_covar_adj.fit(
    verbose=True,
    early_stopping=False,
    mc_adjust=None
)
(
    mad_covar_adj.plot_ate_path()
    + pn.coord_cartesian(ylim=(-1, 3))
    + pn.geom_hline(yintercept=1.3, linetype="dashed")
    + pn.theme(strip_text_x=pn.element_blank())
).save(
    base_dir / "figures" / "mad_covar_adj_ate_path.png",
    width=4,
    height=3,
    dpi=500
)

# Plot ITEs
mad_covar_adj.plot_ites(arm=1, type="histogram", bins=50)
mad_covar_adj.plot_ites(arm=1, type="boxplot")
mad_covar_adj.plot_ites(arm=1, type="density")

# Compare the two methods
mad_comparison = pd.concat([
    mad.estimates().assign(method="MAD"),
    mad_covar_adj.estimates().assign(method="MAD - Covariate Adjusted")
])
(
    pn.ggplot(
        mad_comparison,
        pn.aes(x="factor(method)", y="ate", ymin="lb", ymax="ub")
    )
    + pn.geom_point()
    + pn.geom_linerange()
    + pn.geom_hline(yintercept=1.3, linetype="dashed", color="red")
    + pn.theme_538()
    + pn.labs(x="", y="ATE (95% CS)")
).save(
    base_dir / "figures" / "mad_covar_adj_ate_comparison.png",
    width=4,
    height=3,
    dpi=500
)

# Binary treatment with Bernoulli distributed rewards -------------------------

def reward_covar_adj(arm: int) -> Tuple[float, pd.DataFrame]:
    ate = {0: 0.3, 1: 0.5}
    # Draw X values randomly (here using standard bernoulli distribution)
    X1 = np.random.binomial(1, 0.3)
    X2 = np.random.binomial(1, 0.5)
    X3 = np.random.binomial(1, 0.7)
    # Get the corresponding ATE from the dictionary
    ate = ate[arm]
    # Compute Y_i using the given model
    mean = 0.05 + ate + 0.3 * X1 + 0.1 * X2 - 0.2 * X3
    Y_i = generator.binomial(n=1, p=mean)
    X_df = pd.DataFrame({"X_1": [X1], "X_2": [X2], "X_3": [X3]})
    return Reward(outcome=float(Y_i), covariates=X_df)

def reward_vanilla(arm: int) -> float:
    return Reward(outcome=reward_covar_adj(arm=arm).outcome)

# Vanilla MAD algorithm for 5000 iterations
mad = MAD(
    # bandit=TSBernoulli(k=2, control=0, reward=reward_vanilla),
    bandit=UCB(k=2, control=0, reward=reward_vanilla),
    alpha=0.05,
    delta=lambda x: 1./(x**0.24),
    t_star=int(2e3)
)
mad.fit(verbose=True, early_stopping=False, mc_adjust=None)
(
    mad.plot_ate_path()
    + pn.coord_cartesian(ylim=(-1, 3))
    + pn.geom_hline(yintercept=0.2, linetype="dashed")
    + pn.theme(strip_text_x=pn.element_blank())
).save(
    base_dir / "figures" / "mad_no_covar_ate_path_bern.png",
    width=4,
    height=3,
    dpi=500
)

# Covariate adjusted MAD algorithm for 2000 iterations
mad_covar_adj = MAD(
    # bandit=TSBernoulli(k=2, control=0, reward=reward_covar_adj),
    bandit=UCB(k=2, control=0, reward=reward_covar_adj),
    alpha=0.05,
    delta=lambda x: 1./(x**0.24),
    t_star=int(2e3),
    model=LogitModel,
    pooled=False,
    n_warmup=50
)
mad_covar_adj.fit(
    verbose=True,
    early_stopping=False,
    mc_adjust=None
)
(
    mad_covar_adj.plot_ate_path()
    + pn.coord_cartesian(ylim=(-1, 3))
    + pn.geom_hline(yintercept=0.2, linetype="dashed")
    + pn.theme(strip_text_x=pn.element_blank())
).save(
    base_dir / "figures" / "mad_covar_adj_ate_path_bern.png",
    width=4,
    height=3,
    dpi=500
)

# Plot ITEs
mad.plot_ites(arm=1, type="histogram", bins=50)
mad_covar_adj.plot_ites(arm=1, type="histogram", bins=50)

# Compare the two methods
mad_comparison = pd.concat([
    mad.estimates().assign(method="MAD"),
    mad_covar_adj.estimates().assign(method="MAD - Covariate Adjusted")
])
(
    pn.ggplot(mad_comparison, pn.aes(x="factor(method)", y="ate", ymin="lb", ymax="ub"))
    + pn.geom_point()
    + pn.geom_linerange()
    + pn.geom_hline(yintercept=0.2, linetype="dashed", color="red")
    + pn.theme_538()
    + pn.labs(x="", y="ATE (95% CS)")
).save(
    base_dir / "figures" / "mad_covar_adj_ate_comparison_bern.png",
    width=4,
    height=3,
    dpi=500
)

# Multi-armed treatment example -----------------------------------------------

def reward_covar_adj(arm: int) -> tuple[float, pd.DataFrame]:
    """
    Draws a reward (Y_i) and covariates (X_i) where:
      - Some X_i come from skewed distributions (Lognormal, Exponential, Gamma).
      - The outcome mean function includes nonlinear terms.
      - Different arms have distinct ATE offsets, specified in ate_dict.

    Returns:
      (Y_i, X_df):
        Y_i: float, the reward drawn from a Normal distribution with some nonlinear mean.
        X_df: DataFrame with the covariates for this single observation.
    """
    # Dictionary of arm-specific ATE offsets.
    ate_dict = {
        0: 0.0,
        1: 0.1,
        2: 0.2,
        3: 0.3,
        4: 0.4,
        5: 0.5,
        6: 0.6,
        7: 0.7,
        8: 0.8,
        9: 0.9
    }
    ate_val = ate_dict.get(arm, 0.0)

    # -- Draw some covariates from skewed / non-Gaussian distributions --
    X1 = generator.lognormal(mean=0.5, sigma=1.0)     # Lognormal
    X2 = generator.exponential(scale=1.5)             # Exponential
    X3 = generator.gamma(shape=2.0, scale=2.0)        # Gamma
    X4 = generator.binomial(n=1, p=0.3)               # Binary
    X5 = generator.normal(loc=0.0, scale=1.0)         # Keep a normal variable

    # -- Define a nonlinear mean function that uses these covariates --
    # Example of combining polynomials, logs, and sinusoids:
    mean = (
        0.5
        + ate_val
        + 0.3 * (X1 ** 0.5)         # sqrt transformation of a lognormal
        + 1.2 * np.log1p(X2)        # log(1 + X2) for the exponential
        + 0.5 * np.sin(X3)          # sinusoid in a Gamma variable
        + 1.0 * X4                  # binary shift
        - 0.3 * (X5 ** 2)           # polynomial term in normal X5
    )

    # Draw the actual reward Y_i from a Normal distribution with stdev=1
    Y_i = generator.normal(loc=mean, scale=1.0)

    # Create a DataFrame of covariates
    X_df = pd.DataFrame({
        "X1": [X1],
        "X2": [X2],
        "X3": [X3],
        "X4": [X4],
        "X5": [X5]
    })

    return Reward(outcome=float(Y_i), covariates=X_df)

def reward_vanilla(arm: int) -> float:
    return Reward(outcome=reward_covar_adj(arm=arm).outcome)

# Vanilla MAD algorithm for 5000 iterations
mad = MAD(
    # bandit=TSNormal(k=10, control=0, reward=reward_vanilla),
    bandit=UCB(k=10, control=0, reward=reward_vanilla),
    alpha=0.05,
    delta=lambda x: 1./(x**0.24),
    t_star=int(10e3)
)
mad.fit(verbose=True, early_stopping=False, mc_adjust=None)

# Covariate adjusted MAD for 5000 iterations
mad_covar_adj = MAD(
    # bandit=TSNormal(k=10, control=0, reward=reward_covar_adj),
    bandit=UCB(k=10, control=0, reward=reward_covar_adj),
    alpha=0.05,
    delta=lambda x: 1./(x**0.24),
    t_star=int(10e3),
    model=OLSModel,
    pooled=True,
    n_warmup=1
)
mad_covar_adj.fit(
    verbose=True,
    early_stopping=False,
    mc_adjust=None
)
(
    mad_covar_adj.plot_ate_path()
    + pn.geom_hline(
        pn.aes(yintercept="truth"),
        data=(
            mad_covar_adj
            .estimates()
            .assign(truth=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        ),
        linetype = "dashed"
    )
    + pn.coord_cartesian(ylim=(-2, 3))
)

# Compare the two methods
mad_comparison = pd.concat([
    mad.estimates().assign(method="MAD"),
    mad_covar_adj.estimates().assign(method="Cov. Adjusted")
])
(
    pn.ggplot(mad_comparison, pn.aes(x="factor(method)", y="ate", ymin="lb", ymax="ub"))
    + pn.geom_point()
    + pn.geom_linerange()
    + pn.geom_hline(
        pn.aes(yintercept="truth"),
        data=mad_covar_adj.estimates().assign(truth=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        linetype = "dashed"
    )
    + pn.geom_hline(yintercept=0, linetype="dotted", color="red")
    + pn.facet_wrap("~ arm", nrow=1, labeller=lambda x: f"Arm {x}")
    + pn.theme_538()
    + pn.theme(axis_text_x=pn.element_text(angle=90))
    + pn.labs(x="", y="ATE (95% CS)")
).save(
    base_dir / "figures" / "mad_covar_adj_ate_comparison_many.png",
    width=6,
    height=4,
    dpi=500
)

# Type 1 error simulations ----------------------------------------------------

def compare_type1_error(i, reward, t_star, verbose=False):
    # No multiple comparison adjustment
    mad_no_adjust = MADCovariateAdjusted(
        bandit=UCB(k=10, control=0, reward=reward),
        model=OLSModel,
        pooled=False,
        n_warmup=50,
        alpha=0.05,
        delta=lambda x: 1. / (x ** 0.24),
        t_star=t_star
    )
    mad_no_adjust.fit(verbose=verbose, early_stopping=False, mc_adjust=None)
    # Bonferroni adjustment to ensure FWER <= alpha
    mad_bonferroni = MADCovariateAdjusted(
        bandit=UCB(k=10, control=0, reward=reward),
        model=OLSModel,
        pooled=False,
        n_warmup=50,
        alpha=0.05,
        delta=lambda x: 1. / (x ** 0.24),
        t_star=t_star
    )
    mad_bonferroni.fit(verbose=verbose, early_stopping=False)

    type1_error = pd.concat([
        pd.DataFrame({
            "arm": [k]*2,
            "error": [
                mad_no_adjust._stat_sig_counter[k] > 0,
                mad_bonferroni._stat_sig_counter[k] > 0
            ],
            "adjustment_method": ["None", "Bonferroni"],
            "method": ["MADMod"]*2,
            "idx": [i]*2
        })
        for k in range(1, 10)
    ]).reset_index(drop=True)
    
    return type1_error

def reward_fn(arm: int) -> Tuple[float, pd.DataFrame]:
    ate = {
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0,
        6: 0.0,
        7: 0.0,
        8: 0.0,
        9: 0.0
    }
    # Draw X values randomly (here using standard normal distribution)
    X1 = np.random.randn()
    X2 = np.random.randn()
    X3 = np.random.randn()
    # Get the corresponding ATE from the dictionary
    ate = ate[arm]
    # Compute Y_i using the given model
    mean = 0.5 + ate + 0.3 * X1 + 1.0 * X2 - 0.5 * X3
    Y_i = generator.normal(mean, 1)
    X_df = pd.DataFrame({"X_1": [X1], "X_2": [X2], "X_3": [X3]})
    return Reward(outcome=float(Y_i), covariates=X_df)

type1_error_sim = [
    x for x in
    joblib.Parallel(return_as="generator", n_jobs=-1)(
        joblib.delayed(compare_type1_error)(i, reward=reward_fn, t_star=int(1e4)) for i in range(300)
    )
]

# Calculate the Type 1 Family-wise error rate (FWER)
# aggregate across simulations
overall_type1_error = (
    pd
    .concat(type1_error_sim, ignore_index=True)
    .groupby(["idx", "method", "adjustment_method"], as_index=False)
    .agg(idx_error=("error", "any"))
    .groupby(["method", "adjustment_method"], as_index=False)
    .agg(
        coverage=("idx_error", "mean"),
        n=("idx_error", "count")
    )
)
overall_type1_error["se"] = (
    np.sqrt(overall_type1_error["coverage"]
    * (1 - overall_type1_error["coverage"])
    / overall_type1_error["n"])
)
overall_type1_error["ci_lower"] = (
    overall_type1_error["coverage"]
    - 1.96 * overall_type1_error["se"]
)
overall_type1_error["ci_upper"] = (
    overall_type1_error["coverage"]
    + 1.96 * overall_type1_error["se"]
)
overall_type1_error["coverage_type"] = "Simultaneous"

# Calculate marginal coverage of true parameters
individual_type1_error = (
    pd.concat(type1_error_sim, ignore_index=True)
    .groupby(["method", "adjustment_method"], as_index=False)
    .agg(coverage=("error", "mean"), n=("error", "count"))
)
individual_type1_error["se"] = (
    np.sqrt(individual_type1_error["coverage"]
    * (1 - individual_type1_error["coverage"])
    / individual_type1_error["n"])
)
individual_type1_error["ci_lower"] = (
    individual_type1_error["coverage"]
    - 1.96 * individual_type1_error["se"]
)
individual_type1_error["ci_upper"] = (
    individual_type1_error["coverage"]
    + 1.96 * individual_type1_error["se"]
)
individual_type1_error["coverage_type"] = "Marginal"

# Merge coverage dfs
coverage_df = pd.concat([overall_type1_error, individual_type1_error])

# Plot coverage error
alpha = 0.05
(
    pn.ggplot(
        coverage_df,
        pn.aes(
            x="factor(adjustment_method)",
            y="coverage",
            ymin="ci_lower",
            ymax="ci_upper",
            color="factor(coverage_type)"
        )
    )
    + pn.geom_point(position=pn.position_dodge(width=0.2))
    + pn.geom_linerange(position=pn.position_dodge(width=0.2))
    + pn.geom_hline(yintercept=alpha, linetype="dashed")
    + pn.theme_538()
    + pn.labs(
        x="Adjustment",
        y="Type 1 (coverage) error rate",
        color="Coverage"
    )
).save(
    base_dir / "figures" / "mad_covar_adj_type1_error.png",
    width=6,
    height=4,
    dpi=500
)
