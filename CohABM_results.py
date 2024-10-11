from CohABM_0_2 import CoherenceModel
import pandas as pd
import bnlearn as bn
import mesa
from multiprocessing import freeze_support
from CohABM_BNs import big_sprinkler, common_prior_sprinkler, common_prior_limited_sprinkler

big_sprinkler = big_sprinkler()
common_prior_s = common_prior_sprinkler()
common_prior_limited_s = common_prior_limited_sprinkler()
sprinkler = [("Cloudy", "Sprinkler"), ("Cloudy", "Rain"), ("Rain", "Wet_Grass"), ("Sprinkler", "Wet_Grass")]
limited_sprinkler = [("Rain", "Wet_Grass"), ("Sprinkler", "Wet_Grass")]


params_common = {"N":10,
           "network": ["complete", "wheel", "cycle"],
           "BN": "sprinkler",
           "pulls":100,
           "agent_type":["CoherenceAgent","NormalAgent"],
           "noise":[0, 0.25, 0.5, 0.75],
           "coherence_style":["ogPlus", "og", "shogenji"],
           "misleading_type": ["noisy_data", "big_sprinkler"],
           "prior": ["common"],
           "background": ["sprinkler"]}

params_true = {"N":10,
           "network": ["complete", "wheel", "cycle"],
           "BN": "sprinkler",
           "pulls":100,
           "agent_type":["CoherenceAgent", "NormalAgent"],
           "noise":[0.25, 0.5, 0.75],
           "coherence_style":["ogPlus", "og", "shogenji"],
           "misleading_type": ["noisy_data", "big_sprinkler"],
           "prior": ["true"],
           "background": ["sprinkler"]
}

params_small_world = {"N":10,
           "network": ["complete", "wheel", "cycle"],
           "BN": "sprinkler",
           "pulls":100,
           "agent_type":["CoherenceAgent", "NormalAgent"],
           "noise":[0, 0.25, 0.5, 0.75],
           "coherence_style":["ogPlus", "og", "shogenji"],
           "misleading_type": ["noisy_data", "big_sprinkler"],
           "prior": ["limited_common"],
           "background": ["limited_sprinkler"]}

if __name__ == '__main__':
    freeze_support()
    results = mesa.batch_run(
        CoherenceModel,
        parameters=params_common,
        iterations=20,
        max_steps=30,
        number_processes=None,
        data_collection_period=1,
        display_progress=True,)

        #Export results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results.csv")