import bnlearn as bn
from pgmpy.factors.discrete import TabularCPD
import logging

# Suppressing warning messages from the pgmpy library
logging.getLogger("pgmpy").setLevel(logging.CRITICAL)


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

