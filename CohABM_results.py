from CohABM_0_2 import CoherenceModel
import pandas as pd
import mesa
from multiprocessing import freeze_support


params = {"N":10,
           "network": ["complete", "wheel", "cycle"],
           "BN": "sprinkler",
           "pulls":10,
           "agent_type":["CoherenceAgent", "NormalAgent"],
           "noise":[0.2, 0.5, 0.8],
           "coherence_style":["shogenji", "og", "ogPlus"],
           "misleading_type": ["small_sample", "big_sprinkler"]}

if __name__ == '__main__':
    freeze_support()
    results = mesa.batch_run(
        CoherenceModel,
        parameters=params,
        iterations=30,
        max_steps=30,
        number_processes=None,
        data_collection_period=1,
        display_progress=True,)

#Export results
    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv")