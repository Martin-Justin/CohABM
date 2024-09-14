from CohABM_0_2 import CoherenceModel
import pandas as pd
import mesa
from multiprocessing import freeze_support


params = {"N":10,
           "network": ["complete", "cycle", "wheel"],
           "BN": "sprinkler",
           "pulls":100,
           "agent_type":["CoherenceAgent", "NormalAgent"],
           "noise":0.1,
           "coherence_style":["ogPlus"],
           "misleading_type": ["noisy_data", "big_sprinkler"],
           "prior_type": ["common", "random"]}

if __name__ == '__main__':
    freeze_support()
    results = mesa.batch_run(
        CoherenceModel,
        parameters=params,
        iterations=10,
        max_steps=30,
        number_processes=None,
        data_collection_period=1,
        display_progress=True,)

#Export results
    results_df = pd.DataFrame(results)
    results_df.to_csv("C:/Users/marti/Documents/coherence_results/test_00.csv")