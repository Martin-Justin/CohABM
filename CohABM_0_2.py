"""
 * Authors:   Borut Trpin and Martin Justin
 * Created:   Sept. 9, 2024
 """
import mesa
import numpy as np
import pandas as pd
import random
import bnlearn as bn
import networkx as nx
from pgmpy.factors.discrete import TabularCPD
import logging

# Suppressing warning messages from the pgmpy library
logging.getLogger("pgmpy").setLevel(logging.CRITICAL)



def learn_distribution(DAG, info, method):  # info must be type DataFrame
    return bn.parameter_learning.fit(DAG, info, methodtype=method, verbose=0)



def prob_distr(model):
    nodes = list(model["model"].nodes)
    inference = bn.inference.fit(model, variables=nodes, evidence={}, verbose=0)
    inference_Df=bn.query2df(inference,verbose=0)
    if sum(inference_Df["p"])!=1:
        inference_Df['p']=inference_Df['p']/inference_Df['p'].sum()
        # normalize for the strange cases where the sum doesn't sum up to 1.
    return inference_Df


def marginal_probability(variable, truth, model):
    ### truth is 0/1 for falsehood/truth.
    # Nejc: Fix beacuse truth is sometimes a pd.Series
    if isinstance(truth, pd.Series):
        truth = truth.iloc[0]
    marginal = bn.query2df(bn.inference.fit(model, variables=[variable], evidence={}, verbose=0), verbose=0)
    if len(marginal) == 2:
        return marginal["p"][truth]
    else:
        if marginal[variable][0] == truth:
            return marginal["p"][0]
        else:
            return 1 - marginal["p"][0]


def union_probability(joint_row,prob_distr):
    # take the target row and then look at the opposite row: P(union) = 1 - P(neg-A1 & neg-A2 & ...)
    nodes=[i for i in list(joint_row.keys()) if i!="p"]
    condition = (
        (prob_distr[nodes[0]] == (1 - joint_row[nodes[0]].values[0]))
    )

    for i in range(1, len(nodes)):
        condition &= (
            (prob_distr[nodes[i]] == (1 - joint_row[nodes[i]].values[0]))
        )

    if condition.any():
        return 1-prob_distr[condition]["p"].values[0]
    else:
        # in some cases, the target row is omitted due to 0 zero probability (i.e., the search for all conditions are False), hence union probability is 1.
        return 1


def coherence(model,style="shogenji"):
    # style: "shogenji" (1999; keynes, 1923) or "og" (olsson, 2001; glass, 2001)
    # shogenji style
    nodes = list(model["model"].nodes)
    inference = bn.inference.fit(model, variables=nodes, evidence={}, verbose=0)
    inference_Df = bn.query2df(inference,verbose=0)
    max_p = inference_Df['p'].max()
    inference_Df.loc[1,"p"]=max_p
    max_probability_rows = inference_Df[inference_Df['p'] == max_p]
    if len(max_probability_rows)>1:
       max_probability_rows = max_probability_rows.iloc[:1]
       # random_row=random.choice(range(len(max_probability_rows)))
       # max_probability_rows=max_probability_rows.iloc[random_row:random_row+1]
        #if there are multiple equiprobable and maximum probability states, go for a uniformly random one (this is a simplification, but it shouldn't have too much of an impact on the results, I hope)

    if style=="shogenji":
        # we identify the most probable world-states
        marginals = [marginal_probability(node, value, model) for node, value in max_probability_rows.items() if
                     node != 'p']
        # we calculate the marginal probabilites for each of these states
        marginals_product = np.product(marginals)
        return float(max_p / marginals_product)

    elif style=="og":
        return max_p/union_probability(max_probability_rows,inference_Df)

    elif style=="ogPlus":
        marginals = [marginal_probability(node, value, model) for node, value in max_probability_rows.items() if
                     node != 'p']
        # we calculate the marginal probabilites for each of these states
        marginals_product = np.product(marginals)
        shogenji=float(max_p / marginals_product)
        union=float(union_probability(max_probability_rows,inference_Df))
        marginalsNeg = [marginal_probability(node, 1-value, model) for node, value in max_probability_rows.items() if
                     node != 'p']
        # we calculate t1 minus the product of negated marginal probabilites for each of these states
        marginals_productNeg = 1- np.product(marginalsNeg)
        return float((max_p / marginals_product)*(marginals_productNeg/union))


def kl_divergence(actual, believed):
    # the function takes the actual model (here ground_truth) and the believed model as input and assesses how much information content differs in the believed from the true model.
    actual_df = prob_distr(actual)
    believed_df = prob_distr(believed)
    labels = actual_df.columns.tolist()
    merged_df = pd.merge(actual_df[labels],
                         believed_df[labels],
                         on=labels[:-1],
                         suffixes=('_actual', '_hypothetical'))

    # Calculate KL Divergence
    # Avoid division by zero by replacing zero probabilities with a small value
    epsilon = 1e-10
    merged_df['p_actual'] = np.clip(merged_df['p_actual'], epsilon, 1)
    merged_df['p_hypothetical'] = np.clip(merged_df['p_hypothetical'], epsilon, 1)

    # KL Divergence: sum(P(x) * log(P(x) / Q(x)))
    return (merged_df['p_actual'] * np.log(merged_df['p_actual'] / merged_df['p_hypothetical'])).sum()


def noisy_data(df: pd.DataFrame, percentage: float) -> pd.DataFrame:
    # Calculate how many values to change
    total_values = df.size
    num_values_to_change = int(total_values * percentage)

    # Flatten the DataFrame to 1D to simplify random selection
    flat_df = df.to_numpy().flatten()

    # Randomly choose indices to swap
    indices_to_change = np.random.choice(flat_df.size, num_values_to_change, replace=False)

    # Swap 0s to 1s and 1s to 0s at the selected indices
    flat_df[indices_to_change] = 1 - flat_df[indices_to_change]

    # Reshape the modified array back into the original DataFrame shape
    swapped_df = pd.DataFrame(flat_df.reshape(df.shape), columns=df.columns)

    return swapped_df

# Creating an alternative, misleading version of Sprinkler Bayesian Net
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

big_sprinkler = big_sprinkler()
common_prior = common_prior_sprinkler()



# Agents who consider social coherence
class CoherenceAgent(mesa.Agent):
    def __init__(self, unique_id, model, BN, pulls, coherence_style, prior_type):
        super().__init__(unique_id, model)
        self.truth = self.model.ground_truth
        self.background = bn.import_DAG(BN, CPD=False, verbose=0)
        self.pulls = pulls
        if prior_type == "random":
            self.info = bn.sampling(self.truth, 10, verbose=0)
            self.belief = learn_distribution(self.background, self.info, "ml")
        if prior_type == "common":
            self.belief = common_prior
            self.info = None
        # Possible optimization: create a fixed size dataframe and just fill it and empty it
        self.new_info = None
        self.coherence_style = coherence_style
        self.coherence = coherence(self.belief, self.coherence_style)
        self.accuracy = kl_divergence(self.truth, self.belief)


    # Gathering and sharing data from the world
    def test(self):
        # Agents can receive two different kinds of misleading evidence
        # First, a percentage of values in the sample can randomly change
        if self.model.misleading_type == "noisy_data":
            sample = noisy_data(pd.DataFrame(bn.sampling(self.truth, self.pulls, verbose=0)), self.model.noise)

        # Second, they can draw samples from a different Baysian net
        if self.model.misleading_type == "big_sprinkler":
            if random.random() > self.model.noise:
                sample = pd.DataFrame(bn.sampling(self.truth, self.pulls, verbose=0))
            else:
                sample = pd.DataFrame(bn.sampling(big_sprinkler, self.pulls, verbose=0))

        # Agents add the sample to a DataFrame that (will) also contain(s) their neighbors shared samples
        if self.new_info is not None:
            self.new_info = pd.concat([self.new_info, sample], ignore_index=True)
        else:
            self.new_info = sample

        # Agents share data with their neighbors (within one degree of separation)
        neighbors = self.model.space.get_neighbors(self.pos, False, 1)
        for other_agent in self.model.schedule.agents:
            # Nejc: Refactor for transparency
            if other_agent.unique_id != self.unique_id and other_agent in neighbors:
                if other_agent.new_info is not None:
                    other_agent.new_info = pd.concat([other_agent.new_info, sample], ignore_index=True)
                else:
                    other_agent.new_info = self.new_info


    # This function is used to update agents beliefs about the world
    def update(self):
        if self.info is not None:
            info = pd.concat([self.info, self.new_info], ignore_index=True)
        else:
            info = self.new_info
        # Calculate new belief and its coherence based on aggregated new and existing data
        posterior = learn_distribution(self.background, info, "ml")
        new_coherence = coherence(posterior,self.coherence_style)
        # Check if new belief is more coherent
        # if yes, update the belief, info, coherence and accuracy
        # if not, leave them as they were
        if new_coherence > self.coherence:
            self.info = info
            self.coherence = new_coherence
            self.belief = posterior
            self.accuracy = kl_divergence(self.truth, self.belief)
        # clear new info
        self.new_info = None



# Agents who ignore coherence
class NormalAgent(mesa.Agent):
    def __init__(self, unique_id, model, BN, pulls, prior_type):
        super().__init__(unique_id, model)
        self.truth = self.model.ground_truth
        self.background = bn.import_DAG(BN, CPD=False, verbose=0)
        self.pulls = pulls
        self.info = bn.sampling(self.truth, 10, verbose=0)
        if prior_type == "random":
            self.info = bn.sampling(self.truth, 10, verbose=0)
            self.belief = learn_distribution(self.background, self.info, "ml")
        if prior_type == "common":
            self.belief = common_prior
            self.info = None
        # Nejc: Worth refactoring to create a fixed size dataframe and just fill it and empty it
        self.new_info = None
        self.accuracy = kl_divergence(self.truth, self.belief)
        self.coherence = None


    def test(self):
        # Agents can receive two different kinds of misleading evidence
        # First, a percentage of values in the sample can randomly change
        if self.model.misleading_type == "noisy_data":
            sample = noisy_data(pd.DataFrame(bn.sampling(self.truth, self.pulls, verbose=0)), self.model.noise)

        # Second, they can draw samples from a different Baysian net
        if self.model.misleading_type == "big_sprinkler":
            if random.random() > self.model.noise:
                sample = pd.DataFrame(bn.sampling(self.truth, self.pulls, verbose=0))
            else:
                sample = pd.DataFrame(bn.sampling(big_sprinkler, self.pulls, verbose=0))

        # Agents add the sample to a DataFrame that (will) also contain(s) their neighbors shared samples
        if self.new_info is not None:
            self.new_info = pd.concat([self.new_info, sample], ignore_index=True)
        else:
            self.new_info = sample

        neighbors = self.model.space.get_neighbors(self.pos, False, 1)
        for other_agent in self.model.schedule.agents:
            if other_agent.unique_id != self.unique_id and other_agent in neighbors:
                if other_agent.new_info is not None:
                    other_agent.new_info = pd.concat([other_agent.new_info, sample], ignore_index=True)
                else:
                    other_agent.new_info = self.new_info


    def update(self):
        info = pd.concat([self.info, self.new_info], ignore_index=True)
        self.belief = learn_distribution(self.background, info, "ml")
        self.info = info
        self.accuracy = kl_divergence(self.truth, self.belief)
        self.new_info = None



class CoherenceModel(mesa.Model):
    def __init__(self, N, network, BN, pulls, agent_type, noise, coherence_style, misleading_type, prior_type):
        super().__init__()
        self.num_agents = N
        self.schedule = mesa.time.StagedActivation(self, stage_list=["test", "update"], shuffle=True)
        networks = {"cycle": nx.cycle_graph,
                    "complete": nx.complete_graph,
                    "wheel": nx.wheel_graph}
        self.space = mesa.space.NetworkGrid(networks[network](N))  # Create a model space based on a selected network
        self.ground_truth = bn.import_DAG(BN, CPD=True, verbose=0) # BN with true distribution agents try to approximate
        self.misleading_type = misleading_type
        self.noise = noise

        # Create a list of agents and place them in a network
        for i in range(self.num_agents):
            if agent_type == "CoherenceAgent":
                a = CoherenceAgent(i, self, BN, pulls, coherence_style, prior_type)
            elif agent_type == "NormalAgent":
                a = NormalAgent(i, self, BN, pulls, prior_type)

            self.schedule.add(a)
            self.space.place_agent(a, i) if self.space.is_cell_empty(i) else exit(1)

        # Define data collection -> here we are interested in coherence and accuracy of agents' beliefs
        self.datacollector = mesa.DataCollector(
            agent_reporters={"Coherence": "coherence", "Accuracy": "accuracy"})


    # Run the simulation is steps for all agents
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
