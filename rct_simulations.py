from pathlib import Path
base_dir = Path(__file__).resolve().parent

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List

from src.bandit import AB, TSBernoulli
from src.mad import MAD, MADModified
from src.utils import last

generator = np.random.default_rng(seed=123)

# Import RCT data
rct_data = pd.read_csv(
    base_dir / "data" / "rct_sim" / "SDC - Data - Recoded.csv"
)
outcomes = {
    "PA": "Partisan Animosity",
    "ADA": "Support for Undemocratic Practices",
    "SPV": "Support for Partisan Violence"
}
interventions = dict(enumerate([
    "Null_Control",
    "Alternative_Control",
    "Befriending_Meditation",
    "Chatbot_Quiz",
    "Civity_Storytelling",
    "Common_Identity",
    "Contact_Project",
    "Counterfactual_Selves",
    "Democratic_Fear",
    "Economic_Interests",
    "Empathy_Beliefs",
    "Epistemic_Rescue",
    "Harmful_Experiences",
    "Inparty_Elites",
    "Learning_Goals",
    "Media_Trust",
    "Misperception_Competition",
    "Misperception_Democratic",
    "Misperception_Film",
    "Misperception_Suffering",
    "Moral_Differences",
    "Outparty_Friendship",
    "Partisan_Threat",
    "Party_Overlap",
    "System_Justification",
    "Utah_Cues",
    "Violence_Efficacy"
]))

class SimData:
    def __init__(
        self,
        outcomes: List[str],
        interventions: List[str],
        data: pd.DataFrame
    ):
        self._data = {}
        for outcome in outcomes:
            for intervention in interventions:
                outcome_data = (
                    data
                    .dropna(subset=[outcome])
                    .filter(
                        items=[
                            outcome,
                            "Condition",
                            "Gender",
                            "Age",
                            "Race",
                            "Education",
                            "Inparty_Person",
                            "PI_Pre",
                            "Supplier"
                        ],
                        axis=1
                    )
                )
                intervention_data = (
                    outcome_data[outcome_data["Condition"] == intervention]
                )
                self._data[f"{outcome}_{intervention}"] = intervention_data
    
    def get_data(self, outcome: str, intervention: str):
        """
        Return data for a specific Intervention x Outcome combination.
        """
        return self._data[f"{outcome}_{intervention}"]
    
    def sample_data(self, outcome: str, intervention: str):
        """
        Sample a random row from the data corresponding to the specified
        outcome and intervention combination. This will allow us to effectively
        bootstrap outcome x intervention data.
        """
        data = self.get_data(outcome=outcome, intervention=intervention)
        row = data.sample(n=1)
        return row

# Simulate experiment

## RCT data
sim_data = SimData(
    outcomes=outcomes,
    interventions=list(interventions.values()),
    data=rct_data
)

## MADMod object
def reward_fn(arm: int) -> float:
    row = sim_data.sample_data(outcome="PA", intervention=interventions[arm])
    return row["PA"].iloc[0]

mad_modified = MADModified(
    # bandit=TSBernoulli(
    #     k=len(interventions),
    #     control=0,
    #     reward=reward_fn,
    #     optimize="min"
    # ),
    bandit=AB(k=len(interventions), control=0, reward=reward_fn),
    alpha=0.05,
    delta=lambda x: 1./(x**0.24),
    t_star=int(60e3),
    decay=lambda x: 1./x
)
mad_modified.fit(cs_precision=0.1, mc_adjust=None)
