from CohABM_0_3 import CoherenceModel
import pandas as pd
import mesa
from multiprocessing import freeze_support

params = {"N":1,
           "network": ["complete"],
           "BN": "asia",
           "pulls":100,
           "agent_type":["CoherenceAgent", "NormalAgent"],
           "noise":[0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
           "coherence_style":["ogPlus", "og", "shogenji"],
           "misleading_type": ["noisy_data", "big_asia"],
           "prior": ["approx_true"],
           "background": ["asia"],
           "distance_from_truth": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
          "experts_ratio": 0,
          "latitude":[0],
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
    results_df.to_csv("asia.csv")
