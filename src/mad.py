from .bandit import Bandit
import numpy as np
import pandas as pd
import plotnine as pn
from tqdm import tqdm
from typing import Callable
from .utils import (
    check_shrinkage_rate,
    cs_radius,
    ite,
    last,
    var,
    weighted_probs
)

generator = np.random.default_rng(seed=123)

class MADBase:
    """
    No need to read through the code for this class!!! It just adds some
    methods for plotting the results and spitting out a nicely formatted
    summary. The meat of the algorithm is in the `MAD` class below.
    """
    def __init__(self, bandit: Bandit, alpha: float, delta: Callable[[int], float], t_star: int):
        self._alpha = alpha
        self._ate = []
        self._bandit = bandit
        self._cs_radius = []
        self._cs_width = []
        self._cs_width_benchmark = []
        self._delta = delta
        self._eliminated = {k: False for k in range(bandit.k())}
        self._is_stat_sig = {k: False for k in range(bandit.k())}
        self._ite = []
        self._ite_var = []
        self._n = []
        self._probs = {k: np.empty(0) for k in range(bandit.k())}
        self._rewards = np.empty(0)
        self._stat_sig_counter = []
        self._t_star = t_star
        for _ in range(bandit.k()):
            self._ite.append([])
            self._ite_var.append([])
            self._ate.append([])
            self._cs_radius.append([])
            self._cs_width.append(0)
            self._cs_width_benchmark.append(0)
            self._n.append([0])
            self._stat_sig_counter.append(0)
    
    def estimates(self) -> pd.DataFrame:
        """
        A dataframe of treatment effect estimates and confidence bands
        """
        results = {"arm": [], "ate": [], "lb": [], "ub": []}
        for arm in range(len(self._ate)):
            if arm == self._bandit.control():
                continue
            ate = last(self._ate[arm])
            radius = last(self._cs_radius[arm])
            lb = ate - radius
            ub = ate + radius
            results["arm"].append(arm)
            results["ate"].append(ate)
            results["lb"].append(lb)
            results["ub"].append(ub)
        return pd.DataFrame(results)

    def fit(
        self,
        early_stopping: bool = True,
        cs_precision: float = 0.1,
        verbose: bool = True
    ) -> None:
        """
        Fit the full MAD algorithm for the full time horizon or until there are
        no treatment arms remaining
        """
        if verbose:
            iter_range = tqdm(range(self._t_star), total = self._t_star)
        else:
            iter_range = range(self._t_star)
        for _ in iter_range:
            self.pull(early_stopping=early_stopping, cs_precision=cs_precision)
            # If all treatment arms have been eliminated, end the algorithm
            if early_stopping and all(
                value
                for key, value in self._eliminated.items()
                if key != self._bandit.control()
            ):
                if verbose: 
                    print("Stopping early!")
                break

    def plot_ate(self) -> pn.ggplot:
        """
        Plot the ATEs and CSs for each arm at the end of the experiment
        """
        estimates_df = self.estimates()
        plt = (
            pn.ggplot(
                estimates_df,
                pn.aes(x="factor(arm)", y="ate", ymin="lb", ymax="ub")
            )
            + pn.geom_point()
            + pn.geom_errorbar(width=0.001)
            + pn.labs(x="Arm", y="ATE")
            + pn.theme_538()
        )
        return plt
    
    def plot_ate_path(self) -> pn.ggplot:
        """
        Plot the ATE and CS paths for each arm of the experiment
        """
        arms = list(range(len(self._ate)))
        arms.remove(self._bandit.control())
        estimates = []
        for arm in arms:
            ates = self._ate[arm]
            radii = self._cs_radius[arm]
            ubs = np.nan_to_num([x + y for (x, y) in zip(ates, radii)], nan=np.inf)
            lbs = np.nan_to_num([x - y for (x, y) in zip(ates, radii)], nan=-np.inf)
            estimates_df = pd.DataFrame({
                "arm": [arm]*len(ates),
                "ate": ates,
                "lb": lbs,
                "ub": ubs,
                "t": range(1, len(ates) + 1)
            })
            estimates.append(estimates_df)
        estimates = (
            pd
            .concat(estimates, axis=0)
            .reset_index(drop=True)
            .dropna(subset="ate")
        )
        plt = (
            pn.ggplot(
                data=estimates,
                mapping=pn.aes(
                    x="t",
                    y="ate",
                    ymin="lb",
                    ymax="ub",
                    color="factor(arm)",
                    fill="factor(arm)"
                )
            )
            + pn.geom_line(size=0.3, alpha = 0.8)
            + pn.geom_ribbon(alpha=0.05)
            + pn.facet_wrap(
                "~ arm",
                ncol=2,
                labeller=pn.labeller(arm=lambda v: f"Arm {v}")
            )
            + pn.theme_538()
            + pn.theme(legend_position="none")
            + pn.labs(y="ATE", color="Arm", fill="Arm")
        )
        return plt
    
    def plot_n(self) -> pn.ggplot:
        """
        Plot the total sample size (N) in each arm
        """
        sample_df = pd.DataFrame({
            k: [last(self._n[k])] for k in range(self._bandit.k())
        })
        sample_df = pd.melt(sample_df, value_name="n", var_name="arm")
        plt = (
            pn.ggplot(sample_df, pn.aes(x="factor(arm)", y="n"))
            + pn.geom_bar(stat="identity")
            + pn.labs(x="Arm", y="N")
            + pn.theme_538()
        )
        return plt
    
    def plot_probabilities(self) -> pn.ggplot:
        """
        Plot the arm assignment probabilities across time
        """
        probs_df = pd.melt(
            frame=pd.DataFrame(self._probs),
            value_name="probability",
            var_name="arm"
        )
        probs_df["t"] = probs_df.groupby("arm").cumcount() + 1
        plt = (
            pn.ggplot(
                probs_df,
                pn.aes(
                    x="t",
                    y="probability",
                    color="factor(arm)",
                    group="factor(arm)"
                )
            )
            + pn.geom_line()
            + pn.theme_538()
            + pn.labs(x="t", y="Probability", color="Arm")
        )
        return plt
    
    def plot_sample(self) -> pn.ggplot:
        """
        Plot sample assignment to arms across time
        """
        sample_assignment = pd.concat([
            pd.DataFrame({
                "arm": [arm]*len(self._n[arm]),
                "t": np.array(range(len(self._n[arm]))),
                "n": self._n[arm]
            }) for arm in range(len(self._ate))
        ])
        plt = (
            pn.ggplot(
                data=sample_assignment,
                mapping=pn.aes(
                    x="t", y="n", color="factor(arm)", group="factor(arm)"
                )
            )
            + pn.geom_line()
            + pn.facet_wrap(
                "~ arm",
                ncol=2,
                labeller=pn.labeller(arm=lambda v: f"Arm {v}")
            )
            + pn.theme_538()
            + pn.theme(legend_position="none")
            + pn.labs(y="N", color="Arm", fill="Arm")
        )
        return plt
    
    def summary(self) -> None:
        """
        Print a summary of treatment effect estimates and confidence bands
        """
        print("Treatment effect estimates:")
        for arm in range(len(self._ate)):
            if arm == self._bandit.control():
                continue
            ate = last(self._ate[arm])
            radius = last(self._cs_radius[arm])
            lb = ate - radius
            ub = ate + radius
            print(f"- Arm {arm}: {round(ate, 3)} ({round(lb, 5)}, {round(ub, 5)})")


class MAD(MADBase):
    """
    A class implementing Liang and Bojinov's Mixture-Adaptive Design (MAD).
    
    Parameters
    ----------
    bandit : Bandit 
        This object must implement several crucial
        methods/attributes. For more details on how to create a custom Bandit
        object, see the documentation of the Bandit class.
    alpha : float
        The size of the statistical test (testing for non-zero treatment effects)
    delta : Callable[[int], float]
        A function that generates the real-valued sequence delta_t in Liang
        and Bojinov (Definition 4 - Mixture Adaptive Design). This sequence
        should converge to 0 slower than 1/t^(1/4) where t denotes the time
        frame in {0, ... n}. This function should intake an integer (t) and
        output a float (the corresponding delta_t)
    t_star : int
        The time-step at which we want to optimize the CSs to be tightest.
        E.g. Liang and Bojinov set this to the max horizon of their experiment
    """
    def __init__(
        self,
        bandit: Bandit,
        alpha: float,
        delta: Callable[[int], float],
        t_star: int
    ):
        super().__init__(
            bandit=bandit,
            alpha=alpha,
            delta=delta,
            t_star=t_star
        )
    
    def pull(self, early_stopping: bool = True, cs_precision: float = 0.1) -> None:
        """
        Perform one full iteration of the MAD algorithm.

        Parameters
        ----------
        early_stopping : bool
            Whether or not to stop the experiment early when all the arms have
            statistically significant ATEs.
        cs_precision: float
            This parameter controls how precise we want to make our Confidence
            Sequences (CSs). If `cs_precision = 0` then the experiment will stop
            immediately as soon as all arms are statistically significant.
            If `cs_precision = 0.2` then the experiment will run until all
            CSs are at least 20% tighter (shorter) than they
            were when they became statistically significant. If
            `cs_precision = 0.4` the experiment will run until all CSs are at
            least 40% tighter, and so on.
        """
        # The index of the control arm of the bandit, typically 0
        control = self._bandit.control()
        # The number of bandit arms
        k = self._bandit.k()
        # The CURRENT time step of the bandit
        t = self._bandit.t()
        # The random exploration mixing rate at time step t; For more details
        # on this mixing sequence delta_t look at Liang and Bojinov on Page 13
        # directly above Theorem 1 and in the first new paragraph on Page 11.
        d_t = self._delta(t)
        # This function just ensures that the mixing rate d_t follows the
        # requirements stated in the paper. Can't shrink faster than 1/(t^(1/4)).
        check_shrinkage_rate(t, d_t)
        # Get the arm assignment probabilities from the underlying bandit algo
        arm_probs = self._bandit.probabilities()
        # For each arm, calculate the MAD probabilities based on the bandit probs.
        # This equation is Definition 4 in the paper and it's multi-arm corollary
        # is defined in the third paragraph in Appendix C (Page 34). The
        # multi-class definition is what I'm using here (just a generalization
        # of the 2-class case).
        probs = [d_t/k + (1 - d_t)*p for p in arm_probs.values()]
        # Record these probabilities for plotting later
        for key, value in enumerate(probs):
            self._probs[key] = np.append(self._probs[key], value)
        # Then select the arm as a draw from multinomial with these probabilities
        selected_index = generator.multinomial(1, pvals=probs).argmax()
        selected_arm = list(arm_probs.keys())[selected_index]
        # This is not essential for the MAD algorithm. I'm simply tracking how
        # much sample has been assigned to each arm.
        for arm in range(len(self._ate)):
            if arm == selected_arm:
                self._n[arm].append((last(self._n[arm]) + 1))
            else:
                self._n[arm].append((last(self._n[arm])))
        # Propensity score; obviously just the probability of the selected arm
        propensity = probs[selected_index]
        # True reward resulting from the selected arm
        reward = self._bandit.reward(selected_arm)
        # Record the observed reward
        self._rewards = np.append(self._rewards, reward)
        # Calculate the individual treatment effect estimate (ITE). This is effectively
        # just (reward / propensity). See `utils.ite()` for exactly what it's
        # doing.
        treat_effect = ite(reward, int(selected_arm != control), propensity)
        # Calculate the plug-in variance estimate of the ITE. See `utils.var()`.
        treat_effect_var = var(reward, propensity)
        # Record the ITE and it's variance for calculating the ATE later
        self._ite[selected_arm].append(treat_effect)
        self._ite_var[selected_arm].append(treat_effect_var)
        # Now, for each arm, calculate its ATE and corresponding Confidence Sequence (CS)
        # value for the current time step t.
        for arm in arm_probs.keys():
            # Don't calculate the ATE for the control arm
            if arm == control:
                continue
            else:
                # Get all the ITEs and their variance estimates for the current arm
                # as well as the control arm
                ites = self._ite[control] + self._ite[arm]
                vars = self._ite_var[control] + self._ite_var[arm]
                assert len(ites) == len(vars), "Mismatch in dimensions of ITEs and Variances"
            # If there aren't enough ITEs to calculate the ATE just mark the ATE
            # as missing (np.nan) and mark the CS as infinite
            if len(self._ite[arm]) < 1 or len(self._ite[control]) < 1:
                avg_treat_effect = np.nan
                conf_seq_radius = np.inf
            else:
                # Calculate the ATE from the ITEs:
                # Calculating np.mean(ites) effectively ignores any time periods
                # where neither the current arm nor the control arm were selected.
                # This is WRONG! For each time period (t) where neither the current
                # arm nor the control were selected, the unbiased ATE estimator
                # sets ITE_t = 0. This is why the denominator is the full number
                # of time steps, t.
                avg_treat_effect = np.sum(ites).astype(float)/t
                # The Confidence Sequence calculation is similar. Like the ATE estimator,
                # the estimated variance for any (t) where neither the current arm
                # nor the control arm were selected is just 0. The CS calculation simply
                # sums the estimated variances which is why we only need the non-zero
                # variance estimates. Check out `utils.cs_radius()` for calc details.
                conf_seq_radius = cs_radius(vars, t, self._t_star, self._alpha)
            # Record the ATE and CS for the current arm
            self._ate[arm].append(avg_treat_effect)
            self._cs_radius[arm].append(conf_seq_radius)
            self._cs_width[arm] = 2.0*conf_seq_radius
            # This isn't really the MAD design. This just controls early stopping
            # once all the arms are eliminated (aka all arms have significant ATEs).
            if early_stopping and arm != control:
                # If the arm's ATE is undefined (insufficient sample size) skip
                if np.isnan(avg_treat_effect) or np.isinf(conf_seq_radius):
                    continue
                # Is the arm statistically significant?
                stat_sig = np.logical_or(
                    0 <= avg_treat_effect - conf_seq_radius,
                    0 >= avg_treat_effect + conf_seq_radius
                )
                if stat_sig:
                    # Mark arm as statistically significant
                    self._is_stat_sig[arm] = True
                    # If the CS is statistically significant for the first time
                    # we will attempt to increase precision by decreasing the
                    # interval width by X% relative to its current width
                    self._stat_sig_counter[arm] += 1
                    if self._stat_sig_counter[arm] == 1:
                        self._cs_width_benchmark[arm] = self._cs_width[arm]
                    # Now, eliminate the arm iff it is <= (1-X)% of it's width when
                    # initially marked as statistically significant
                    threshold = (1-cs_precision)*self._cs_width_benchmark[arm]
                    if self._cs_width[arm] <= threshold:
                        self._eliminated[arm] = True
                else:
                    # Otherwise mark it as NOT significant
                    self._is_stat_sig[arm] = False
                    # If the arm has been significant but is not any more,
                    # un-eliminate the arm
                    if self._eliminated[arm]:
                        self._eliminated[arm] = False
        return None

class MADModified(MADBase):
    """
    This class implements my minor (but important) changes to the MAD design.
    All the code in the `pull()` method is identical to that in the MAD class
    above. I've only documented the chunks where the code is different, which
    is where I implement my modifications to the MAD algorithm.
    
    Parameters
    ----------
    bandit : Bandit 
        This object must implement several crucial
        methods/attributes. For more details on how to create a custom Bandit
        object, see the documentation of the Bandit class.
    alpha : float
        The size of the statistical test (testing for non-zero treatment effects)
    delta : Callable[[int], float]
        A time-decreasing function that generates the real-valued sequence
        delta_t in Liang and Bojinov (Definition 4 - Mixture Adaptive Design).
        This sequence should converge to 0 slower than 1/t^(1/4) where t denotes
        the time frame in {0, ... n}. This function should intake an integer (t)
        and output a float (the corresponding delta_t).
    t_star : int
        The time-step at which we want to optimize the CSs to be tightest.
        E.g. Liang and Bojinov set this to the max horizon of their experiment
    decay : Callable[[int], float]
        `decay()` should intake the current time step t and output a value
        in [0, 1] where 1 represents no decay and 0 represents complete decay.
        This function is similar to the `delta()` argument above. However,
        `delta()` determines the amount of random exploration in the MAD
        algorithm at time t. In contrast, `decay()` is a time-decreasing
        function that determines how quickly the bandit assignment
        probabilities for arm k decay to 0 once arm k's ATE (ATE_k) is
        statistically significant. Setting a constant `decay = lambda _: 1` makes
        this method identical to the vanilla MAD design. In contrast,
        `decay = lambda _: 0` is the same as setting the bandit probabilities
        for arm k to 0 as soon as it has a significant ATE.
    
    Attributes
    ----------
    _weights : Dict[int, float]
        These weights are calculated by `decay()`. Each arm has a weight. When
        an arm has a weight of 1, its bandit assignment probabilities are not
        adjusted. Once an arm is statistically significant, its weight begins
        to decay to 0, and so does its bandit assignment probabilities. As an
        arm's weight decays to 0, it "shifts" its probability onto currently
        under-powered arms. This iterative procedure continues to focus more
        sample on under-powered arms until either all arms have significant ATEs
        or the experiment has ended.
    """
    def __init__(
        self,
        bandit: Bandit,
        alpha: float,
        delta: Callable[[int], float],
        t_star: int,
        decay: Callable[[int], float] = lambda x: 1/np.sqrt(x)
    ):
        super().__init__(
            bandit=bandit,
            alpha=alpha,
            delta=delta,
            t_star=t_star
        )
        self._decay = decay
        self._eliminated_t = {k: None for k in range(bandit.k())}
        self._is_stat_sig = {k: False for k in range(bandit.k())}
        self._weights = {k: 1. for k in range(bandit.k())}

    def pull(self, early_stopping: bool = True, cs_precision: float = 0.1) -> None:
        """
        Perform one full iteration of the modified MAD algorithm
        """
        control = self._bandit.control()
        k = self._bandit.k()
        t = self._bandit.t()
        d_t = self._delta(t)
        check_shrinkage_rate(t, d_t)
        arm_probs = self._bandit.probabilities()

        # Re-weight the bandit probabilities according to each arm's assigned
        # weights. Take a look at `utils.weighted_probs()`; it has a very
        # detailed explanation of how the re-weighting works.
        arm_probs = weighted_probs(arm_probs, self._weights)

        probs = [d_t/k + (1 - d_t)*p for p in arm_probs.values()]
        for key, value in enumerate(probs):
            self._probs[key] = np.append(self._probs[key], value)
        selected_index = generator.multinomial(1, pvals=probs).argmax()
        selected_arm = list(arm_probs.keys())[selected_index]
        for arm in range(len(self._ate)):
            if arm == selected_arm:
                self._n[arm].append((last(self._n[arm]) + 1))
            else:
                self._n[arm].append((last(self._n[arm])))
        propensity = probs[selected_index]
        reward = self._bandit.reward(selected_arm)
        self._rewards = np.append(self._rewards, reward)
        treat_effect = ite(reward, int(selected_arm != control), propensity)
        treat_effect_var = var(reward, propensity)
        self._ite[selected_arm].append(treat_effect)
        self._ite_var[selected_arm].append(treat_effect_var)
        for arm in arm_probs.keys():
            if arm == control:
                ites = self._ite[control]
                vars = self._ite_var[control]
            else:
                # Get the ITE and variance estimates for the current arm and control arm
                ites = self._ite[control] + self._ite[arm]
                vars = self._ite_var[control] + self._ite_var[arm]
                assert len(ites) == len(vars), "Mismatch in dimensions of ITEs and Variances"
            if len(self._ite[arm]) < 1 or len(self._ite[control]) < 1:
                avg_treat_effect = np.nan
                conf_seq_radius = np.inf
            else:
                avg_treat_effect = np.sum(ites).astype(float)/t
                conf_seq_radius = cs_radius(vars, t, self._t_star, self._alpha)
            self._ate[arm].append(avg_treat_effect)
            self._cs_radius[arm].append(conf_seq_radius)
            self._cs_width[arm] = 2.0*conf_seq_radius
            if early_stopping and arm != control:
                if np.isnan(avg_treat_effect) or np.isinf(conf_seq_radius):
                    continue
                stat_sig = np.logical_or(
                    0 <= avg_treat_effect - conf_seq_radius,
                    0 >= avg_treat_effect + conf_seq_radius
                )
                if stat_sig:
                    self._is_stat_sig[arm] = True
                    self._stat_sig_counter[arm] += 1
                    if self._stat_sig_counter[arm] == 1:
                        self._cs_width_benchmark[arm] = self._cs_width[arm]
                    threshold = (1-cs_precision)*self._cs_width_benchmark[arm]
                    if self._cs_width[arm] <= threshold and not self._eliminated[arm]:
                        self._eliminated[arm] = True

                        # We need to know for HOW LONG has the arm been eliminated.
                        # The longer the arm is eliminated (statistically significant)
                        # the closer its corresponding weight gets to 0
                        self._eliminated_t[arm] = t

                else:
                    self._is_stat_sig[arm] = False
                    if self._eliminated[arm]:
                        self._eliminated[arm] = False
                        self._eliminated_t[arm] = None
                
                # Now, update the arm's assigned weight. If the arm has been
                # eliminated this weight will decay to 0. Every time the arm
                # goes from stat. sig. to non-stat. sig., this weight resets
                # to 1 (the default value for non-stat. sig. arms) and restarts
                # its decay path to 0.
                if not self._eliminated[arm]:
                    self._weights[arm] = 1.
                else:
                    self._weights[arm] = np.max([
                        self._decay(t + 1 - self._eliminated_t[arm]),
                        1e-10
                    ]).astype(float)
        return None   