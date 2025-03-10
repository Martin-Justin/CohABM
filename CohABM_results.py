from CohABM_0_3 import CoherenceModel
import pandas as pd
import mesa
from multiprocessing import freeze_support

params = {"N":1,
           "network": ["complete"],
           "BN": "sprinkler",
           "pulls": [100],
           "agent_type":["CoherenceAgent", "NormalAgent"],
           "noise":[0.1],
           "coherence_style":["ogPlus", "og", "shogenji"],
           "misleading_type": ["big_sprinkler"],
           "prior": ["approx_true"],
           "background": ["sprinkler"],
           "distance_from_truth": [0.1],
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
        number_processes=4,
        data_collection_period=1,
        display_progress=True,)

    #Export results
    results_df = pd.DataFrame(results)
    results_df.to_csv("C:/Users/marti/My Drive/Faks/PHD/papers/Social Coherence in Science/testing/new_code_test.csv")
