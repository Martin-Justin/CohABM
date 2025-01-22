from CohABM_0_2 import CoherenceModel
import pandas as pd
import mesa
from multiprocessing import freeze_support
from CohABM_BNs import big_sprinkler, common_prior_sprinkler, common_prior_limited_sprinkler

big_sprinkler = big_sprinkler()
common_prior_s = common_prior_sprinkler()
common_prior_limited_s = common_prior_limited_sprinkler()
sprinkler = [("Cloudy", "Sprinkler"), ("Cloudy", "Rain"), ("Rain", "Wet_Grass"), ("Sprinkler", "Wet_Grass")]
limited_sprinkler = [("Rain", "Wet_Grass"), ("Sprinkler", "Wet_Grass")]


params = {"N":10,
           "network": ["complete", "wheel", "cycle"],
           "BN": "sprinkler",
           "pulls":100,
           "agent_type":["CoherenceAgent","NormalAgent"],
           "noise":[0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
           "coherence_style":["ogPlus", "og", "shogenji"],
           "misleading_type": ["noisy_data"],
           "prior": ["approx_true"],
           "background": ["sprinkler"],
           "distance_from_truth": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
}

if __name__ == '__main__':
    freeze_support()
    results = mesa.batch_run(
        CoherenceModel,
        parameters=params,
        iterations=50,
        max_steps=50,
        number_processes=None,
        data_collection_period=1,
        display_progress=True,)

        #Export results
    results_df = pd.DataFrame(results)
    results_df.to_csv("path")
