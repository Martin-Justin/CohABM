import bnlearn as bn
from pgmpy.factors.discrete import TabularCPD
import logging
import random
import numpy as np
import copy

# Suppressing warning messages from the pgmpy library
logging.getLogger("pgmpy").setLevel(logging.CRITICAL)

def prob_distr(model):
    nodes = list(model["model"].nodes)
    inference = bn.inference.fit(model, variables=nodes, evidence={}, verbose=0)
    inference_Df=bn.query2df(inference,verbose=0)
    if sum(inference_Df["p"])!=1:
        inference_Df['p']=inference_Df['p']/inference_Df['p'].sum()
        # normalize for the strange cases where the sum doesn't sum up to 1.
    return inference_Df

def apply_random_change(values, noise_level):
    modified_values = []

    for n in values:
        # Define the range for x
        lower_bound = max(0, n - noise_level)  # Don't go below 0
        upper_bound = min(1, n + noise_level)  # Don't go above 1

        # Generate a random change x within the bounds
        x = np.random.uniform(lower_bound, upper_bound)

        # Apply the change and add to the modified list
        modified_values.append(x)
    return modified_values


def noisier_cpd(cpdTable, noise_level):
    vals = cpdTable.values.reshape(2, -1).copy()
    vals[0] = apply_random_change(vals[0], noise_level)
    vals[1] = [1 - x for x in vals[0]]
    return vals


def noisier_model(model, noise_level):
    model = copy.deepcopy(model)
    structure = list(model["model"].edges)
    cpds = model["model"].cpds
    new_cpds = []
    for cpd in cpds:
        noisy_values = noisier_cpd(cpd, noise_level)

        # Get evidence and evidence_card from the CPD
        evidence = cpd.variables[1:]  # Exclude the variable itself
        evidence_card = cpd.cardinality[1:]  # Exclude the variable itself

        # Reconstruct the TabularCPD
        new_cpd = TabularCPD(
            variable=cpd.variable,
            variable_card=cpd.cardinality[0],  # Cardinality of the main variable
            values=noisy_values,
            evidence=evidence if evidence else None,  # None if no evidence
            evidence_card=evidence_card if evidence else None,
        )
        new_cpds.append(new_cpd)
    return bn.make_DAG(structure, CPD=new_cpds, verbose=0)


def big_sprinkler():
    sprinkler_list = [("Cloudy", "Sprinkler"), ("Cloudy", "Rain"), ("Rain", "Wet_Grass"), ("Sprinkler", "Wet_Grass")]
    cpd_Cloudy = TabularCPD(variable='Cloudy', variable_card=2, values=[[0.5], [0.5]])

    # CPD for Sprinkler: P(Sprinkler | Cloudy)
    cpd_Sprinkler = TabularCPD(variable='Sprinkler', variable_card=2,
                               values=[[0.5, 0.9],  # P(Sprinkler=0 | Cloudy=0), P(Sprinkler=0 | Cloudy=1)
                                       [0.5, 0.1]],  # P(Sprinkler=1 | Cloudy=0), P(Sprinkler=1 | Cloudy=1)
                               evidence=['Cloudy'], evidence_card=[2])

    # CPD for Rain: P(Rain | Cloudy)
    cpd_Rain = TabularCPD(variable='Rain', variable_card=2,
                          values=[[0.8, 0.2],  # P(Rain=0 | Cloudy=0), P(Rain=0 | Cloudy=1)
                                  [0.2, 0.8]],  # P(Rain=1 | Cloudy=0), P(Rain=1 | Cloudy=1)
                          evidence=['Cloudy'], evidence_card=[2])

    # CPD for Wet_Grass: P(Wet_Grass | Sprinkler, Rain)
    cpd_Wet_Grass = TabularCPD(variable='Wet_Grass', variable_card=2,
                               values=[[0.99, 0.9, 0.1, 0.01],
                                       # P(Wet_Grass=0 | Sprinkler=0, Rain=0), ..., P(Wet_Grass=0 | Sprinkler=1, Rain=1)
                                       [0.01, 0.1, 0.9, 0.99]],
                               # P(Wet_Grass=1 | Sprinkler=0, Rain=0), ..., P(Wet_Grass=1 | Sprinkler=1, Rain=1)
                               evidence=['Sprinkler', 'Rain'], evidence_card=[2, 2])

    return bn.make_DAG(sprinkler_list, CPD=[cpd_Cloudy, cpd_Sprinkler, cpd_Rain, cpd_Wet_Grass], verbose=0)

def common_prior_sprinkler():
    sprinkler_list = [("Cloudy", "Sprinkler"), ("Cloudy", "Rain"), ("Rain", "Wet_Grass"), ("Sprinkler", "Wet_Grass")]
    cpd_Cloudy = TabularCPD(variable='Cloudy', variable_card=2, values=[[0.5], [0.5]])

    # CPD for Sprinkler: P(Sprinkler | Cloudy)
    cpd_Sprinkler = TabularCPD(variable='Sprinkler', variable_card=2,
                               values=[[0.5, 0.5],  # P(Sprinkler=0 | Cloudy=0), P(Sprinkler=0 | Cloudy=1)
                                       [0.5, 0.5]],  # P(Sprinkler=1 | Cloudy=0), P(Sprinkler=1 | Cloudy=1)
                               evidence=['Cloudy'], evidence_card=[2])

    # CPD for Rain: P(Rain | Cloudy)
    cpd_Rain = TabularCPD(variable='Rain', variable_card=2,
                          values=[[0.5, 0.5],  # P(Rain=0 | Cloudy=0), P(Rain=0 | Cloudy=1)
                                  [0.5, 0.5]],  # P(Rain=1 | Cloudy=0), P(Rain=1 | Cloudy=1)
                          evidence=['Cloudy'], evidence_card=[2])

    # CPD for Wet_Grass: P(Wet_Grass | Sprinkler, Rain)
    cpd_Wet_Grass = TabularCPD(variable='Wet_Grass', variable_card=2,
                               values=[[0.5, 0.5, 0.5, 0.5],
                                       # P(Wet_Grass=0 | Sprinkler=0, Rain=0), ..., P(Wet_Grass=0 | Sprinkler=1, Rain=1)
                                       [0.5, 0.5, 0.5, 0.5]],
                               # P(Wet_Grass=1 | Sprinkler=0, Rain=0), ..., P(Wet_Grass=1 | Sprinkler=1, Rain=1)
                               evidence=['Sprinkler', 'Rain'], evidence_card=[2, 2])

    return bn.make_DAG(sprinkler_list, CPD=[cpd_Cloudy, cpd_Sprinkler, cpd_Rain, cpd_Wet_Grass], verbose=0)


def common_prior_limited_sprinkler():
    sprinkler_list = [("Rain", "Wet_Grass"), ("Sprinkler", "Wet_Grass")]

    # CPD for Sprinkler: P(Sprinkler | Cloudy)
    cpd_Sprinkler = TabularCPD(variable='Sprinkler', variable_card=2, values=[[0.5], [0.5]])

    # CPD for Rain: P(Rain | Cloudy)
    cpd_Rain = TabularCPD(variable='Rain', variable_card=2, values=[[0.5], [0.5]])

    # CPD for Wet_Grass: P(Wet_Grass | Sprinkler, Rain)
    cpd_Wet_Grass = TabularCPD(variable='Wet_Grass', variable_card=2,
                               values=[[0.5, 0.5, 0.5, 0.5],
                                       # P(Wet_Grass=0 | Sprinkler=0, Rain=0), ..., P(Wet_Grass=0 | Sprinkler=1, Rain=1)
                                       [0.5, 0.5, 0.5, 0.5]],
                               # P(Wet_Grass=1 | Sprinkler=0, Rain=0), ..., P(Wet_Grass=1 | Sprinkler=1, Rain=1)
                               evidence=['Sprinkler', 'Rain'], evidence_card=[2, 2])

    return bn.make_DAG(sprinkler_list, CPD=[cpd_Sprinkler, cpd_Rain, cpd_Wet_Grass], verbose=0)

# def limited_sprinkler():
#     lim_sprinker_list = [("Rain", "Wet_Grass"), ("Sprinkler", "Wet_Grass")]
#     return bn.make_DAG(lim_sprinker_list, CPD=None, verbose=0)

