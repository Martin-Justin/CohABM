"""
 * Authors:   Borut Trpin and Martin Justin
 * Created:   Sept. 9, 2024
 """
import mesa
import pandas as pd
import random
import bnlearn as bn
import networkx as nx
import logging

from CohABM_functions import learn_distribution, kl_divergence, noisy_data, coherence, noisier_model

# Suppressing warning messages from the pgmpy library
logging.getLogger("pgmpy").setLevel(logging.CRITICAL)


# Agents who ignore coherence
class NormalAgent(mesa.Agent):
    def __init__(self, unique_id, model, pulls, prior, background, distance_from_truth):
        super().__init__(unique_id, model)
        self.truth = self.model.ground_truth
        backgrounds = {"sprinkler": [("Cloudy", "Sprinkler"), ("Cloudy", "Rain"), ("Rain", "Wet_Grass"),
                                     ("Sprinkler", "Wet_Grass")],
                       "limited_sprinkler": [("Rain", "Wet_Grass"), ("Sprinkler", "Wet_Grass")],
                       "asia": [('asia', 'tub'), ('tub', 'either'), ('smoke', 'lung'), ('smoke', 'bronc'), ('lung', 'either'), ('bronc', 'dysp'), ('either', 'xray'), ('either', 'dysp')]}
        self.distance_from_truth = distance_from_truth
        self.background = bn.make_DAG(backgrounds[background], CPD=None, verbose=0)
        self.pulls = pulls
        self.edges = set()
        for pairs in backgrounds[background]:
            for edge in pairs:
                self.edges.add(edge)
        if prior == "random":
            self.info = bn.sampling(self.truth, 10, verbose=0)
            info_edges = set(self.info.columns)
            section = info_edges - (info_edges & self.edges)
            if section:
                self.info.drop(section, axis="Columns", inplace=True)
            self.belief = learn_distribution(self.background, self.info, "ml")
        elif prior == "approx_true":
            self.belief = noisier_model(self.truth, self.distance_from_truth)
            self.info = bn.sampling(self.belief, self.pulls, verbose=0)
        elif prior == "true":
            self.belief = self.truth
            self.info = bn.sampling(self.belief, self.pulls, verbose=0)
        elif prior == "common":
            self.belief = bn.import_DAG("common_prior_sprinkler.bif", CPD=True, verbose=0)
            self.info = bn.sampling(self.belief, self.pulls, verbose=0)
        elif prior == "limited_common":
            self.belief = bn.import_DAG("common_prior_limited_sprinkler.bif", CPD=True, verbose=0)
            self.info = bn.sampling(self.belief, self.pulls, verbose=0)
        # Nejc: Worth refactoring to create a fixed size dataframe and just fill it and empty it
        self.new_info = None
        self.accuracy = kl_divergence(self.truth, self.belief)
        self.coherence = None
        # if self.accuracy > 1:
        #     self.noise = 0.49
        # else:
        #     self.noise = self.model.noise * self.accuracy

    def test(self):
        # Agents can receive two different kinds of misleading evidence
        # First, a percentage of values in the sample can randomly change
        if self.model.misleading_type == "noisy_data":
            sample = noisy_data(pd.DataFrame(bn.sampling(self.truth, self.pulls, verbose=0)), self.model.noise)

        elif self.model.misleading_type == "combination":
            if random.random() > self.model.noise:
                sample = noisy_data(pd.DataFrame(bn.sampling(self.truth, self.pulls, verbose=0)), self.model.noise)
            else:
                sample = noisy_data(pd.DataFrame(bn.sampling(self.model.misleading_type, self.pulls, verbose=0)), self.model.noise)

        else:
            if random.random() > self.model.noise:
                sample = pd.DataFrame(bn.sampling(self.truth, self.pulls, verbose=0))
            else:
                sample = pd.DataFrame(bn.sampling(self.model.misleading_type, self.pulls, verbose=0))

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
        # Agents can have a limited picture of the world, so they can get more evidence that they can account of
        # This first checks if info includes more edges than agents' background
        # Then it appropriately limits the evidence if necessary
        info_edges = set(self.new_info.columns)
        section = info_edges - (info_edges & self.edges)
        if section:
            self.new_info.drop(section, axis="columns", inplace=True)

        if self.info is not None:
            info = pd.concat([self.info, self.new_info], ignore_index=True)
        else:
            info = self.new_info
        self.belief = learn_distribution(self.background, info, "ml")
        self.info = info
        self.accuracy = kl_divergence(self.truth, self.belief)
        self.new_info = None


class CoherenceAgent(NormalAgent):
    def __init__(self, unique_id, model, pulls, coherence_style, prior, background, distance_from_truth):
        super().__init__(unique_id, model, pulls, prior, background, distance_from_truth)
        self.coherence_style = coherence_style
        self.coherence = coherence(self.belief, self.coherence_style)

    # This function is used to update agents beliefs about the world
    def update(self):
        # Agents can have a limited picture of the world, so they can get more evidence that they can account of
        # This first checks if info includes more edges than agents' background
        # Then it appropriately limits the evidence if necessary
        info_edges = set(self.new_info.columns)
        section = info_edges - (info_edges & self.edges)
        if section:
            self.new_info.drop(section, axis="columns", inplace=True)

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


class ModerateCoherenceAgent(NormalAgent):
    def __init__(self, unique_id, model, pulls, coherence_style, prior, background, distance_from_truth, latitude):
        super().__init__(unique_id, model, pulls, prior, background, distance_from_truth)
        self.coherence_style = coherence_style
        self.coherence = coherence(self.belief, self.coherence_style)
        self.latitude = latitude
        self.evidence_counter = 0

    def update(self):
        info_edges = set(self.new_info.columns)
        section = info_edges - (info_edges & self.edges)
        if section:
            self.new_info.drop(section, axis="columns", inplace=True)

        if self.info is not None:
            info = pd.concat([self.info, self.new_info], ignore_index=True)
        else:
            info = self.new_info
        posterior = learn_distribution(self.background, info, "ml")
        new_coherence = coherence(posterior,self.coherence_style)
        if self.evidence_counter < self.latitude and (self.coherence-new_coherence) < (self.coherence * ((self.latitude - self.evidence_counter)/100)):
            self.info = info
            self.coherence = new_coherence
            self.belief = posterior
            self.accuracy = kl_divergence(self.truth, self.belief)
          # self.evidence_counter += 1
        else:
            if new_coherence > self.coherence:
                self.info = info
                self.coherence = new_coherence
                self.belief = posterior
                self.accuracy = kl_divergence(self.truth, self.belief)
             #  self.evidence_counter += 1
        self.new_info = None

class CoherenceModel(mesa.Model):
    def __init__(self, N, network, BN, pulls, agent_type, noise, coherence_style, misleading_type, prior, background, distance_from_truth, experts_ratio, latitude):
        super().__init__()
        self.num_agents = N
        self.schedule = mesa.time.StagedActivation(self, stage_list=["test", "update"], shuffle=True)
        networks = {"cycle": nx.cycle_graph,
                    "complete": nx.complete_graph,
                    "wheel": nx.wheel_graph}
        self.space = mesa.space.NetworkGrid(networks[network](N))  # Create a model space based on a selected network
        self.ground_truth = bn.import_DAG(f"{BN}.bif", CPD=True, verbose=0) # BN with true distribution agents try to approximate
        if misleading_type == "noisy_data":
            self.misleading_type = "noisy_data"
        elif misleading_type == "combination":
            self.misleading_type = bn.import_DAG(f"big_sprinkler_minus.bif", CPD=True, verbose=0)
        else:
            self.misleading_type = bn.import_DAG(f"{misleading_type}.bif", CPD=True, verbose=0)
        self.noise = noise

        # Create a list of agents and place them in a network
        experts = N * experts_ratio
        agents = list()
        for i in range(self.num_agents):
            if agent_type == "CoherenceAgent":
                if i < experts:
                    a = CoherenceAgent(i, self, pulls, coherence_style, prior, background, 0)
                else:
                    a = CoherenceAgent(i, self, pulls, coherence_style, prior, background, distance_from_truth)
            elif agent_type == "NormalAgent":
                if i < experts:
                    a = NormalAgent(i, self, pulls, prior, background, 0)
                else:
                    a = NormalAgent(i, self, pulls, prior, background, distance_from_truth)
            elif agent_type == "ModerateCoherenceAgent":
                if i < experts:
                    a = ModerateCoherenceAgent(i, self, pulls, coherence_style, prior, background, 0, latitude)
                else:
                    a = ModerateCoherenceAgent(i, self, pulls, coherence_style, prior, background, distance_from_truth, latitude)

            self.schedule.add(a)
            agents.append(a)

        positions = list(range(self.num_agents))
        random.shuffle(positions)
        for agent, position in zip(agents, positions):
            self.space.place_agent(agent, position) if self.space.is_cell_empty(position) else exit(1)

        # Define data collection -> here we are interested in coherence and accuracy of agents' beliefs
        self.datacollector = mesa.DataCollector(
            agent_reporters={"Coherence": "coherence", "Accuracy": "accuracy"})


    # Run the simulation is steps for all agents
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
