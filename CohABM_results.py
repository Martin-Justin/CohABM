from CohABM_0_3 import CoherenceModel
import pandas as pd
import mesa
from multiprocessing import freeze_support

params = {"N":10,
           "network": ["complete"],
           "BN": "sprinkler",
           "pulls":10,
           "agent_type":["CoherenceAgent", "NormalAgent"],
           "noise":[0.3],
           "coherence_style":["ogPlus", "og", "shogenji"],
           "misleading_type": ["noisy_data"],
           "prior": ["approx_true"],
           "background": ["sprinkler"],
           "distance_from_truth": 0.1,
          "experts_ratio": 0,
}

if __name__ == '__main__':
    freeze_support()
    results = mesa.batch_run(
        CoherenceModel,
        parameters=params,
        iterations=30,
        max_steps=50,
        number_processes=None,
       data_collection_period=1,
        display_progress=True,)

    #Export results
    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv")