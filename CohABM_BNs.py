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
                               values=[[0.99, 0.6, 0.1, 0.01],
                                       # P(Wet_Grass=0 | Sprinkler=0, Rain=0), ..., P(Wet_Grass=0 | Sprinkler=1, Rain=1)
                                       [0.01, 0.4, 0.9, 0.99]],
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

def big_asia():
    asia_list = [('asia', 'tub'), ('tub', 'either'), ('smoke', 'lung'), ('smoke', 'bronc'), ('lung', 'either'),
             ('bronc', 'dysp'), ('either', 'xray'), ('either', 'dysp')]

    # Ignore the comments !!!!
    cpd_asia = TabularCPD(variable="asia", variable_card=2,
                          values=[[0.01],  # P(Asia=1)
                                  [0.99]]) # P(Asia=0)

    # CPD for Tuberculosis given Asia (column order fixed)
    cpd_tuberculosis = TabularCPD(variable="tub", variable_card=2,
                                  values=[[0.05, 0.01],  # P(Tub=1 | Asia=0, Asia=1)
                                          [0.95, 0.99]], # P(Tub=0 | Asia=0, Asia=1)
                                  evidence=["asia"], evidence_card=[2])

    # CPD for Smoking (root node)
    cpd_smoking = TabularCPD(variable="smoke", variable_card=2,
                             values=[[0.5],   # P(Smoking=1)
                                     [0.5]])  # P(Smoking=0)

    # Changed so that smoking does not casues cancer
    cpd_lung_cancer = TabularCPD(variable="lung", variable_card=2,
                                 values=[[0.011, 0.01],  # P(Lung=1 | Smoking=1, Smoking=0)
                                         [0.999, 0.99]], # P(Lung=0 | Smoking=1 , Smoking=0)
                                 evidence=["smoke"], evidence_card=[2])

    # CPD for Bronchitis given Smoking (column order fixed)
    cpd_bronchitis = TabularCPD(variable="bronc", variable_card=2,
                                values=[[0.31, 0.30],  # P(Bronch=1 | Smoking=1, Smoking=0)
                                        [0.69, 0.70]], # P(Bronch=0 | Smoking=1, Smoking=0)
                                evidence=["smoke"], evidence_card=[2])

    # CPD for Either (TubOrCancer) given Tuberculosis and Lung Cancer (column order fixed)
    cpd_either = TabularCPD(variable="either", variable_card=2,
                            values=[[1., 1., 1., 0.],  # P(Either=1 | Lung=0, Tub=0; Lung=0, Tub=1; Lung=1, Tub=0; Lung=1, Tub=1)
                                    [0., 0., 0., 1.]], # P(Either=0 | Lung=0, Tub=0; Lung=0, Tub=1; Lung=1, Tub=0; Lung=1, Tub=1)
                            evidence=["lung", "tub"],
                            evidence_card=[2, 2])

    # CPD for XRay given Either (column order fixed)
    cpd_xray = TabularCPD(variable="xray", variable_card=2,
                          values=[[0.98, 0.05],  # P(XRay=1 | Either=0, Either=1)
                                  [0.02, 0.95]], # P(XRay=0 | Either=0, Either=1)
                          evidence=["either"], evidence_card=[2])

    # CPD for Dyspnea given Bronchitis and Either (column order fixed)
    cpd_dyspnea = TabularCPD(variable="dysp", variable_card=2,
                             values=[[0.90, 0.80, 0.70, 0.10],
                                     [0.10, 0.20, 0.30, 0.90]],
                             evidence=["bronc", "either"],
                             evidence_card=[2, 2])


    cpds = [cpd_asia, cpd_bronchitis, cpd_dyspnea, cpd_either, cpd_lung_cancer, cpd_smoking, cpd_tuberculosis, cpd_xray]

    return bn.make_DAG(asia_list, CPD=cpds, verbose=0)

# def limited_sprinkler():
#     lim_sprinker_list = [("Rain", "Wet_Grass"), ("Sprinkler", "Wet_Grass")]
#     return bn.make_DAG(lim_sprinker_list, CPD=None, verbose=0)

