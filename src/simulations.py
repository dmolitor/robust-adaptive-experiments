from pathlib import Path
import sys

if sys.stdin.isatty():
    import os
    os.chdir("./src/")
    base_dir = Path().resolve().parent
else:
    base_dir = Path(__file__).resolve().parent.parent

from bandit import TSBernoulli
from mad import MAD, MADMolitor
import numpy as np
import pandas as pd
import plotnine as pn
from rct import rct
from utils import last

generator = np.random.default_rng(seed=123)

# Simple (binary) example! ----------------------------------------------------

def reward_fn(arm: int) -> float:
    values = {
        0: generator.binomial(1, 0.5),
        1: generator.binomial(1, 0.6)
    }
    return values[arm]

# MAD algorithm
print("Fitting binary MAD simulation ...")
exp_simple = MAD(
    bandit=TSBernoulli(k=2, control=0, reward=reward_fn),
    alpha=0.05,
    delta=lambda x: 1./(x**0.24),
    t_star=int(20e3)
)
# NOTE: cs_precision = 0 says "quit as soon as stat. sig."
exp_simple.fit(cs_precision=0)

# RCT with the same sample size
print("Fitting binary RCT simulation ...")
rct_results = rct(
    t=exp_simple._bandit._t,
    k=len(exp_simple._ate),
    control=0,
    reward_fn=reward_fn,
    alpha=0.05
)

# Plot convergence of CSs around ATE
print("Plotting CSs for binary simulation ...")
(
    (
        exp_simple.plot()
        + pn.coord_cartesian(ylim=(-.5, 1.5))
        + pn.geom_hline(
            mapping=pn.aes(yintercept="ate", color="factor(arm)"),
            data=pd.DataFrame({"arm": list(range(1, 2)), "ate": [0.1]}),
            linetype="dotted"
        )
        + pn.theme(strip_text=pn.element_blank()) 
    )
    .save(
        filename=base_dir / "figures" / "simple_ate_cs.png",
        width=5,
        height=4,
        dpi=300
    )
)

# Combine ATE estimates and plot
print("Plotting ATEs for binary simulation ...")
ate_df = pd.concat(
    [
        exp_simple.summary(estimates=True).assign(which="mad"),
        rct_results.assign(which="rct"),
        pd.DataFrame({
            "arm": list(range(1, 2)),
            "ate": [0.1],
            "which": ["truth"]
        })
    ],
    axis=0
)
(
    pn.ggplot(
        ate_df,
        mapping=pn.aes(
            x="factor(arm)",
            y="ate",
            ymin="lb",
            ymax="ub",
            color="which"
        )
    )
    + pn.geom_point(position=pn.position_dodge(width=0.2))
    + pn.geom_errorbar(position=pn.position_dodge(width=0.2), width=0.1)
    + pn.theme_538()
    + pn.labs(x = "Arm", y="ATE", color="")
).save(
    base_dir / "figures" / "simple_ate_estimates.png",
    height=4,
    width=3,
    dpi=300
)

# Plot sample assignment
print("Plotting sample assignment for binary simulation ...")
(
    exp_simple.plot_sample()
).save(
    base_dir / "figures" / "simple_sample_assign.png",
    height=3,
    width=5,
    dpi=300
)

# More complex example! -------------------------------------------------------

def reward_fn(arm: int) -> float:
    values = {
        0: generator.binomial(1, 0.5),  # Control arm
        1: generator.binomial(1, 0.6),  # ATE = 0.1
        2: generator.binomial(1, 0.62), # ATE = 0.12
        3: generator.binomial(1, 0.8),  # ATE = 0.3
        4: generator.binomial(1, 0.85)  # ATE = 0.35
    }
    return values[arm]

adapt_level = 0.24

## Run a Modified MAD experiment

# MAD design with Molitor adaptation
# NOTE: the decay rate of 1/(t^(1/8)) is pretty slow. This means
# that we want to err on the side of honoring the bandit arm
# assignment probabilities.
print("Fitting Molitor MAD multi-treatment simulation ...")
mad_experiment_molitor = MADMolitor(
    bandit=TSBernoulli(k=5, control=0, reward=reward_fn),
    alpha=0.05,
    delta=lambda x: 1./(x**adapt_level),
    t_star=int(30e3),
    decay=lambda x: 1/(x**(1/8))
)
# cs_precision = 0.1 means "improve CS width by 10% once stat. sig."
mad_experiment_molitor.fit(cs_precision=0.1)

## Now, for comparison, run the vanilla MAD design

print("Fitting pure MAD multi-treatment simulation ...")
mad_experiment = MAD(
    bandit=TSBernoulli(k=5, control=0, reward=reward_fn),
    alpha=0.05,
    delta=lambda x: 1./(x**adapt_level),
    t_star=mad_experiment_molitor._bandit._t
)
mad_experiment.fit(early_stopping=False)

## Plot the ATES and sample assignment from the vanilla MAD experiment

mad_exp_df = pd.concat(
    [
        mad_experiment.summary(estimates=True).assign(which="mad"),
        pd.DataFrame({
            "arm": list(range(1, 5)),
            "ate": [0.1, 0.12, 0.3, 0.35],
            "which": ["truth"]*4
        })
    ],
    axis=0
)
print("Plotting MAD TS ATEs ...")
(
    pn.ggplot(
        mad_exp_df,
        mapping=pn.aes(
            x="factor(arm)",
            y="ate",
            ymin="lb",
            ymax="ub",
            color="which"
        )
    )
    + pn.geom_point(position=pn.position_dodge(width=0.2))
    + pn.geom_errorbar(position=pn.position_dodge(width=0.2), width=0.1)
    + pn.theme_538()
    + pn.labs(x = "Arm", y="ATE", color="")
).save(
    base_dir / "figures" / "mad_ts_ate_estimates.png",
    height=4,
    width=4,
    dpi=300
)

print("Plotting MAD TS sample assignment ...")
(
    mad_experiment.plot_sample()
).save(
    base_dir / "figures" / "mad_ts_assign.png",
    height=3,
    width=4,
    dpi=300
)
print(f"MAD TS with {mad_experiment._t_star} samples")

## Compare the ATEs between the MAD and Modified designs

print("Plotting comparison between MAD ATEs and MAD Modified ATEs ...")
molitor_df = mad_experiment_molitor.summary(estimates=True).assign(which="molitor")
mad_df = mad_experiment.summary(estimates=True).assign(which="mad")
results = pd.concat(
    [
        molitor_df,
        mad_df,
        pd.DataFrame({
            "arm": list(range(1, 5)),
            "ate": [0.1, 0.12, 0.3, 0.35],
            "which": ["truth"]*(5-1)
        })
    ],
    axis=0
)
print("Plotting comparison ATEs for pure MAD and Molitor MAD ...")
(
    pn.ggplot(results, mapping=pn.aes(x="factor(arm)", y="ate", ymin="lb", ymax="ub", color="which"))
    + pn.geom_point(position=pn.position_dodge(width=0.3))
    + pn.geom_errorbar(position=pn.position_dodge(width=0.3), width=0.2)
    + pn.theme_538()
    + pn.labs(y="ATE", color="Experiment")
).save(
    base_dir / "figures" / "mad_adapted_ate_estimates.png",
    height=6,
    width=8,
    dpi=300
)

## Compare sample assignment to arms between the MAD and Modified designs

sample_sizes = pd.concat([
    pd.DataFrame(x) for x in
    [
        {
            "arm": [k for k in range(len(mad_experiment_molitor._ate))],
            "n": [last(n) for n in mad_experiment_molitor._n],
            "which": ["molitor"]*len(mad_experiment_molitor._ate)
        },
        {
            "arm": [k for k in range(len(mad_experiment._ate))],
            "n": [last(n) for n in mad_experiment._n],
            "which": ["mad"]*len(mad_experiment._ate)
        }
    ]
])
print("Plotting sample size comparison ...")
(
    pn.ggplot(sample_sizes, pn.aes(x="factor(arm)", y="n", fill="which", color="which"))
    + pn.geom_bar(stat="identity", position=pn.position_dodge(width=0.75), width=0.7)
    + pn.theme_538()
    + pn.labs(x="Arm", y="N", color="Experiment", fill="Experiment")
).save(
    base_dir / "figures" / "mad_adapted_assign.png",
    height=4,
    width=5,
    dpi=300
)
