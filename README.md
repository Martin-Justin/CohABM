# Background

What grounds our knowledge about the world? One possible answer to this question takes the form of 
coherentism, the idea that knowledge "arises from relations of coherence or mutual support between 
our beliefs" (Olsson, [2022](https://philpapers.org/rec/OLSC-4)). 

Philosophers of science have also suggested that coherence should have important role in scientific 
inquiry. Recently, Goldberg and Khalifa ([2022](https://link.springer.com/article/10.1007/s11098-022-01849-8))
argued for a social role of coherence consideration in science. Specifically, they proposed that for a 
set of a given scientific community C's justified background information B, some belief *p* is unjustified
in C if it negatively coheres with B. 

The idea behind CohABM is to test whether this type of social coherentism can help agents to form accurate
beliefs about the world.

# The model

Agents in the model try to lean the probability distribution of a given Bayesian Network. 

They do this by
taking a number of samples from the distribution and then guessing the distribution based on the samples. 
For example, imagine they try to learn the distribution of the 
"[Sprinkler](https://www.bayesserver.com/examples/networks/sprinkler)" network. A table with 5 samples 
might look like this:

| Cloudy | Rain | Sprinkler | Wet Grass |
|--------|------|-----------|-----------|
| 1      | 1    | 0         | 1         |
| 1      | 1    | 0         | 0         |
| 0      | 0    | 1         | 1         |
| 0      | 0    | 0         | 0         |
| 1      | 1    | 1         | 1         |

Agents can also share their data with their neighbors. They can be connected into different
networks which determine the number of neighbors that they have (for an introduction to network epistemology, 
see Zollman, [2007](https://www.jstor.org/stable/10.1086/525605)). 

There are two different kinds of agents implemented in the model. What we can "Normal agents" simply 
sample, share, and receive evidence, based on which they then update their belief about the world. 


Meanwhile, the other kind of agents, here called "Coherence agents", also check how this new evidence 
affect the coherence of their belief about the world. These agents first sample,share, and receive 
new evidence. But before accepting this new evidence, they check whether their belief about what 
is the most probable world, is more coherent than their previous one. They can do this using three different 
coherence measures, one suggested by Shogenji ([1999](https://philpapers.org/rec/SHOICT-2)), another
one independently presented by Olsson ([2002](https://philpapers.org/rec/OLSWIT)) and 
Glass ([2002](https://link.springer.com/chapter/10.1007/3-540-45750-X_23)), and the third one suggested by
Hartmann and Trpin ([forthcoming](https://philsci-archive.pitt.edu/22792/)).

Agents can also get misleading evidence with some probability ("noise"), which is determined as a 
parameter of the model. 

The performance of agents is calculated using 
[Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
a measure of how much one probability distribution differ from a reference probability distribution.

# Running the model

The libraries needed to run the model can be found in *requirements.txt*. The model was tested 
using Python version `3.9.1`.

The model is best run using `mesa.batchrun()` function. Below is an example run that covers a wide section
of the parameter space of the model.

```
params = {"N": range(2, 10, 1),
           "network": ["complete", "wheel", "cycle"],
           "BN": "sprinkler",
           "pulls":range(10, 100, 10),
           "agent_type":["CoherenceAgent", "NormalAgent"],
           "noise": range(0, 1, 0.1),
           "coherence_style":["shogenji", "og", "ogPlus"],
           "misleading_type": ["small_sample", "big_sprinkler"]}

if __name__ == '__main__':  
    freeze_support()     # Suggested for running multiprocessing on Windows
    results = mesa.batch_run(
        CoherenceModel,
        parameters=params,
        iterations=100,
        max_steps=50,
        number_processes=None,
        data_collection_period=1,
        display_progress=True,)
```

---

Both the code for the model and its conceptual development is a joint work by [Borut Trpin](https://boruttrpin.weebly.com/) 
and [Martin Justin](https://martin-justin.github.io/).

