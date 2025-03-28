from pathlib import Path
base_dir = Path(__file__).resolve().parent

import joblib
import numpy as np
import pandas as pd
import plotnine as pn
from scipy.stats import t

from src.bandit import Reward, TSBernoulli
from src.mad import MAD, MADModified, MADCovariateAdjusted
from src.utils import last

generator = np.random.default_rng(seed=123)

# Reward function for simulations
#
# Simulate a very simple binary experiment
def reward_fn(arm: int) -> float:
    values = {
        0: generator.binomial(1, 0.5),  # Control arm
        1: generator.binomial(1, 0.6),  # ATE = 0.1
    }
    return Reward(outcome=values[arm])

mad = MAD(
    TSBernoulli(k=2, control=0, reward=reward_fn),
    alpha=0.05,
    delta=lambda x: 1./(x**0.24),
    t_star=int(5e3)
)
mad.fit(cs_precision=0.0, mc_adjust=None)
(
    mad.plot_ate_path()
    + pn.coord_cartesian(ylim=(-1, 1.5))
    + pn.geom_hline(yintercept=0.1, linetype="dashed")
    + pn.theme(strip_text_x=pn.element_blank())
).save(
    base_dir / "figures" / "mad_ts_ate.png",
    width=4,
    height=3,
    dpi=500
)
(
    mad.plot_n()
).save(
    base_dir / "figures" / "mad_ts_n.png",
    width=2,
    height=3,
    dpi=500
)

# Reward function for simulations
#
# We demonstrate this with an experiment simulating a control arm and four
# treatment arms with ATEs of 0.1, 0.12, 0.3, and 0.32, respectively, over a
# fixed sample size of 20,000. We expect the bandit algorithm to allocate most of
# the sample to arms 3 and 4, leaving arms 1 and 2 under-powered.
def reward_fn(arm: int) -> float:
    values = {
        0: generator.binomial(1, 0.5),  # Control arm
        1: generator.binomial(1, 0.6),  # ATE = 0.1
        2: generator.binomial(1, 0.62), # ATE = 0.12
        3: generator.binomial(1, 0.8),  # ATE = 0.3
        4: generator.binomial(1, 0.82)  # ATE = 0.32
    }
    return Reward(outcome=values[arm])

# Algorithm comparison
# 
# I compare the two algorithms to highlight the benefits of the modified
# approach. The modified algorithm significantly improves power to detect
# non-zero ATEs in all treatment arms and provides more precise ATE estimates
# than the original MAD algorithm with the same sample size. However, this comes
# at the cost of assigning more sample to sub-optimal arms, where "optimal" is
# defined by the underlying bandit algorithm.

## A comparison with one simulation

# Run the modified algorithm
mad_modified = MADModified(
    bandit=TSBernoulli(k=5, control=0, reward=reward_fn),
    alpha=0.05,
    delta=lambda x: 1./(x**0.24),
    t_star=int(30e3),
    decay=lambda x: 1./(x**(1./8.))
)
mad_modified.fit(
    cs_precision=0.1,
    verbose=True,
    early_stopping=True,
    mc_adjust=None
)

# Run the vanilla algorithm
mad_vanilla = MAD(
    bandit=TSBernoulli(k=5, control=0, reward=reward_fn),
    alpha=0.05,
    delta=lambda x: 1./(x**0.24),
    t_star=mad_modified._bandit._t
)
mad_vanilla.fit(verbose=True, early_stopping=False, mc_adjust=None)

# Plot the MAD vanilla estimates
ates = pd.concat(
    [
        mad_modified.estimates().assign(which="MADMod"),
        mad_vanilla.estimates().assign(which="MAD"),
        pd.DataFrame({
            "arm": list(range(1, 5)),
            "ate": [0.1, 0.12, 0.3, 0.32],
            "which": ["Truth"]*(4)
        })
    ],
    axis=0
)

(
    pn.ggplot(
        ates[ates["which"] != "MADMod"],
        mapping=pn.aes(
            x="factor(arm)",
            y="ate",
            ymin="lb",
            ymax="ub",
            color="which"
        )
    )
    + pn.geom_point(position=pn.position_dodge(width=0.3))
    + pn.geom_errorbar(position=pn.position_dodge(width=0.3), width=0.001)
    + pn.geom_hline(yintercept=0, linetype="dashed", color="black")
    + pn.theme_538()
    + pn.labs(x="Arm", y="ATE", color="Method")
).save(
    base_dir / "figures" / "mad_ts_ate_underpowered.png",
    width=5,
    height=3,
    dpi=500
)

(
    mad_vanilla.plot_n()
).save(
    base_dir / "figures" / "mad_ts_n_underpowered.png",
    width=3,
    height=3,
    dpi=500
)

# Compare the ATEs and CSs
(
    pn.ggplot(
        ates,
        mapping=pn.aes(
            x="factor(arm)",
            y="ate",
            ymin="lb",
            ymax="ub",
            color="which"
        )
    )
    + pn.geom_point(position=pn.position_dodge(width=0.3))
    + pn.geom_errorbar(position=pn.position_dodge(width=0.3), width=0.001)
    + pn.geom_hline(yintercept=0, linetype="dashed", color="black")
    + pn.theme_538()
    + pn.labs(x="Arm", y="ATE", color="Method")
).save(
    base_dir / "figures" / "mad_vs_madmod_ate_underpowered.png",
    width=5,
    height=3,
    dpi=500
)


# And the following plot compares the sample assignment to the treatment arms
# of the two algorithms:
sample_sizes = pd.concat([
    pd.DataFrame(x) for x in
    [
        {
            "arm": [k for k in range(len(mad_modified._ate))],
            "n": [last(n) for n in mad_modified._n],
            "which": ["MADMod"]*len(mad_modified._ate)
        },
        {
            "arm": [k for k in range(len(mad_vanilla._ate))],
            "n": [last(n) for n in mad_vanilla._n],
            "which": ["MAD"]*len(mad_vanilla._ate)
        }
    ]
])
(
    pn.ggplot(sample_sizes, pn.aes(x="factor(arm)", y="n", fill="which", color="which"))
    + pn.geom_bar(stat="identity", position=pn.position_dodge(width=0.75), width=0.7)
    + pn.theme_538()
    + pn.labs(x="Arm", y="N", color="Method", fill="Method")
).save(
    base_dir / "figures" / "mad_vs_madmod_n_underpowered.png",
    width=3,
    height=3,
    dpi=500
)


# Simulation results over 1,0000 runs
# 
# We can more precisely quantify the improvements by running 1,000 simulations,
# comparing Type 2 error and confidence band width between the vanilla MAD
# algorithm and the modified algorithm. Each simulation runs for 20,000
# iterations with early stopping. If the modified algorithm stops early, the
# vanilla algorithm will also stop early to maintain equal sample sizes in each
# simulation.
def compare(i):
    mad_modified = MADModified(
        bandit=TSBernoulli(k=5, control=0, reward=reward_fn),
        alpha=0.05,
        delta=lambda x: 1. / (x ** 0.24),
        t_star=int(3e4),
        decay=lambda x: 1. / (x ** (1. / 8.))
    )
    mad_modified.fit(cs_precision=0.1, verbose=False, early_stopping=True)

    # Run the vanilla algorithm
    mad_vanilla = MAD(
        bandit=TSBernoulli(k=5, control=0, reward=reward_fn),
        alpha=0.05,
        delta=lambda x: 1. / (x ** 0.24),
        t_star=mad_modified._bandit._t
    )
    mad_vanilla.fit(verbose=False, early_stopping=False)

    # Calculate the Type 2 error and the Confidence Sequence width

    ## For modified algorithm
    mad_mod_n = (
        pd
        .DataFrame([
            {"arm": k, "n": last(mad_modified._n[k])}
            for k in range(mad_modified._bandit.k())
            if k != mad_modified._bandit.control()
        ])
        .assign(
            n_pct=lambda x: x["n"].apply(lambda y: y/np.sum(x["n"]))
        )
    )
    mad_mod_df = (
        mad_modified
        .estimates()
        .assign(
            idx=i,
            method="modified",
            width=lambda x: x["ub"] - x["lb"],
            error=lambda x: ((0 > x["lb"]) & (0 < x["ub"]))
        )
        .merge(mad_mod_n, on="arm", how="left")
    )

    ## For vanilla algorithm
    mad_van_n = (
        pd
        .DataFrame([
            {"arm": k, "n": last(mad_vanilla._n[k])}
            for k in range(mad_vanilla._bandit.k())
            if k != mad_vanilla._bandit.control()
        ])
        .assign(
            n_pct=lambda x: x["n"].apply(lambda y: y/np.sum(x["n"]))
        )
    )
    mad_van_df = (
        mad_vanilla
        .estimates()
        .assign(
            idx=i,
            method="mad",
            width=lambda x: x["ub"] - x["lb"],
            error=lambda x: ((0 > x["lb"]) & (0 < x["ub"]))
        )
        .merge(mad_van_n, on="arm", how="left")
    )

    out = {
        "metrics": pd.concat([mad_mod_df, mad_van_df]),
        "reward": {
            "modified": np.sum(mad_modified._rewards),
            "mad": np.sum(mad_vanilla._rewards)
        }
    }
    return out

# Execute in parallel with joblib
comparison_results_list = [
    x for x in
    joblib.Parallel(return_as="generator", n_jobs=-1)(
        joblib.delayed(compare)(i) for i in range(100)
    )
]

# Compare performance on key metrics across simulations
metrics_df = pd.melt(
    (
        pd
        .concat([x["metrics"] for x in comparison_results_list])
        .reset_index(drop=True)
        .assign(error=lambda x: x["error"].apply(lambda y: int(y)))
    ),
    id_vars=["arm", "method"],
    value_vars=["width", "error", "n", "n_pct"],
    var_name="meas",
    value_name="value"
)
metrics_df["method"] = (
    metrics_df["method"]
    .apply(lambda x: {"modified": "MADMod", "mad": "MAD"}[x])
)

# Compare reward accumulation across simulations
reward_df = pd.melt(
    pd.DataFrame([x["reward"] for x in comparison_results_list]),
    value_vars=["modified", "mad"],
    var_name="method",
    value_name="reward"
)
reward_df["method"] = (
    reward_df["method"]
    .apply(lambda x: {"modified": "MADMod", "mad": "MAD"}[x])
)

metrics_summary = (
    metrics_df
    .groupby(["arm", "method", "meas"], as_index=False).agg(
        mean=("value", "mean"),
        std=("value", "std"),
        n=("value", "count")
    )
    .assign(
        se=lambda x: x["std"] / np.sqrt(x["n"]),
        t_val=lambda x: t.ppf(0.975, x["n"] - 1),
        ub=lambda x: x["mean"] + x["t_val"] * x["se"],
        lb=lambda x: x["mean"] - x["t_val"] * x["se"]
    )
    .drop(columns=["se", "t_val"])
)


# The following plot shows the mean (and 95% confidence intervals) of the
# Type 2 error and CS width for both algorithms.
facet_labels = {
    "error": "Type 2 error",
    "width": "Interval width",
    "n": "Sample size",
    "n_pct": "Sample size %"
}
(
    pn.ggplot(
        metrics_summary[metrics_summary["meas"].isin(["error", "width"])],
        pn.aes(
            x="factor(arm)",
            y="mean",
            ymin="lb",
            ymax="ub",
            color="method"
        )
    )
    + pn.geom_point(position=pn.position_dodge(width=0.2), size=0.7)
    + pn.geom_linerange(position=pn.position_dodge(width=0.2))
    + pn.facet_wrap(
        "~ meas",
        labeller=lambda x: facet_labels[x],
        scales="free"
    )
    + pn.theme_538()
    + pn.labs(x="Arm", y="", color="Method")
).save(
    base_dir / "figures" / "ic2s2_figure1.png",
    width=5,
    height=2,
    dpi=500
)

# These plots illustrate the tradeoffs of the modified algorithm. On average,
# it allocates significantly more sample to sub-optimal arms compared to the
# standard MAD algorithm.
(
    pn.ggplot(
        metrics_summary[metrics_summary["meas"].isin(["n", "n_pct"])],
        pn.aes(
            x="factor(arm)",
            y="mean",
            ymin="lb",
            ymax="ub",
            color="method"
        )
    )
    + pn.geom_point(position=pn.position_dodge(width=0.2), size=0.7)
    + pn.geom_linerange(position=pn.position_dodge(width=0.2))
    + pn.facet_wrap(
        "~ meas",
        labeller=lambda x: facet_labels[x],
        scales="free"
    )
    + pn.theme_538()
    + pn.labs(x="Arm", y="", color="Method")
).save(
    base_dir / "figures" / "ic2s2_figure2.png",
    width=5,
    height=2,
    dpi=500
)


# As a result, this reallocation reduces total reward accumulation. The
# difference in accumulated reward across the 1,000 simulations is shown below:
(
    pn.ggplot(reward_df, pn.aes(x="method", y="reward"))
    + pn.geom_boxplot()
    + pn.theme_538()
    + pn.labs(x="Method", y="Cumulative reward")
)

# Type 1 error simulations

def compare_type1_error(i, reward, t_star):
    # No multiple comparison adjustment
    mad_modified = MADModified(
        bandit=TSBernoulli(k=10, control=0, reward=reward),
        alpha=0.05,
        delta=lambda x: 1. / (x ** 0.24),
        t_star=t_star,
        decay=lambda x: 1. / (x ** (1. / 8.))
    )
    mad_modified.fit(verbose=False, early_stopping=False, mc_adjust=None)
    # Bonferroni adjustment to ensure FWER <= alpha
    mad_modified_bonferroni = MADModified(
        bandit=TSBernoulli(k=10, control=0, reward=reward),
        alpha=0.05,
        delta=lambda x: 1. / (x ** 0.24),
        t_star=t_star,
        decay=lambda x: 1. / (x ** (1. / 8.))
    )
    mad_modified_bonferroni.fit(verbose=False, early_stopping=False)

    type1_error = pd.concat([
        pd.DataFrame({
            "arm": [k]*2,
            "error": [
                mad_modified._stat_sig_counter[k] > 0,
                mad_modified_bonferroni._stat_sig_counter[k] > 0
            ],
            "adjustment_method": ["None", "Bonferroni"],
            "method": ["MADMod"]*2,
            "idx": [i]*2
        })
        for k in range(1, 10)
    ]).reset_index(drop=True)
    
    return type1_error

def reward_fn(arm: int) -> float:
    values = {
        0: generator.binomial(1, 0.5),
        1: generator.binomial(1, 0.5),
        2: generator.binomial(1, 0.5),
        3: generator.binomial(1, 0.5),
        4: generator.binomial(1, 0.5),
        5: generator.binomial(1, 0.5),
        6: generator.binomial(1, 0.5),
        7: generator.binomial(1, 0.5),
        8: generator.binomial(1, 0.5),
        9: generator.binomial(1, 0.5)
    }
    return Reward(outcome=values[arm])

type1_error_sim = [
    x for x in
    joblib.Parallel(return_as="generator", n_jobs=-1)(
        joblib.delayed(compare_type1_error)(i, reward=reward_fn, t_star=int(1e4)) for i in range(1000)
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
