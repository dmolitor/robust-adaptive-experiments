from pathlib import Path
base_dir = Path(__file__).resolve().parent

import joblib
import numpy as np
import pandas as pd
import plotnine as pn
from scipy.stats import t
from typing import Dict, Tuple

from src.bandit import TSBernoulli, TSNormal
from src.mad import MAD, MADModified, MADCovariateAdjusted
from src.utils import last

generator = np.random.default_rng(seed=123)

def reward_covar_adj(arm: int) -> Tuple[float, pd.DataFrame]:
    ate = {0: 0.0, 1: 1.0}
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
    return float(Y_i), X_df

def reward_vanilla(arm: int) -> float:
    return reward_covar_adj(arm=arm)[0]

# Vanilla MAD algorithm for 2000 iterations
mad = MAD(
    bandit=TSNormal(k=2, control=0, reward=reward_vanilla),
    alpha=0.05,
    delta=lambda x: 1./(x**0.24),
    t_star=int(2e3)
)
mad.fit(cs_precision=0.1, verbose=True, early_stopping=False, mc_adjust=None)
(
    mad.plot_ate_path()
    + pn.coord_cartesian(ylim=(-1, 3))
    + pn.geom_hline(yintercept=1, linetype="dashed")
    + pn.theme(strip_text_x=pn.element_blank())
).save(
    base_dir / "figures" / "mad_no_covar_ate_path.png",
    width=4,
    height=3,
    dpi=500
)

# Covariate adjusted MAD algorithm for 2000 iterations
mad_covar_adj = MADCovariateAdjusted(
    bandit=TSNormal(k=2, control=0, reward=reward_covar_adj),
    alpha=0.05,
    delta=lambda x: 1./(x**0.1),
    t_star=int(2e3)
)
mad_covar_adj.fit(cs_precision=0.1, verbose=True, early_stopping=False, mc_adjust=None)
(
    mad_covar_adj.plot_ate_path()
    + pn.coord_cartesian(ylim=(-1, 3))
    + pn.geom_hline(yintercept=1, linetype="dashed")
    + pn.theme(strip_text_x=pn.element_blank())
).save(
    base_dir / "figures" / "mad_covar_adj_ate_path.png",
    width=4,
    height=3,
    dpi=500
)

# Compare the two methods
mad_comparison = pd.concat([
    mad.estimates().assign(method="MAD"),
    mad_covar_adj.estimates().assign(method="MAD - Covariate Adjusted")
])
(
    pn.ggplot(mad_comparison, pn.aes(x="factor(method)", y="ate", ymin="lb", ymax="ub"))
    + pn.geom_point()
    + pn.geom_linerange()
    + pn.geom_hline(yintercept=1, linetype="dashed", color="red")
    + pn.theme_538()
    + pn.labs(x="", y="ATE (95% CS)")
).save(
    base_dir / "figures" / "mad_covar_adj_ate_comparison.png",
    width=4,
    height=3,
    dpi=500
)

# Type 1 error simulations

# TODO: currently this shouldn't (doesn't) work. Need to implement AIPW #######
# estimates for multiple arms simultaneously                            #######

def compare_type1_error(i, reward, t_star):
    # No multiple comparison adjustment
    mad_no_adjust = MADCovariateAdjusted(
        bandit=TSBernoulli(k=10, control=0, reward=reward),
        alpha=0.05,
        delta=lambda x: 1. / (x ** 0.24),
        t_star=t_star
    )
    mad_no_adjust.fit(verbose=False, early_stopping=False, mc_adjust=None)
    # Bonferroni adjustment to ensure FWER <= alpha
    mad_bonferroni = MADCovariateAdjusted(
        bandit=TSBernoulli(k=10, control=0, reward=reward),
        alpha=0.05,
        delta=lambda x: 1. / (x ** 0.24),
        t_star=t_star
    )
    mad_bonferroni.fit(verbose=False, early_stopping=False)

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
    return float(Y_i), X_df

type1_error_sim = [
    x for x in
    joblib.Parallel(return_as="generator", n_jobs=-1)(
        joblib.delayed(compare_type1_error)(i, reward=reward_fn, t_star=int(1e4)) for i in range(100)
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
)