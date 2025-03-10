import pandas as pd
import bnlearn as bn
import numpy as np
import copy
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
    ### truth is the index of the state: for binary variables, this means 0 for True, 1 for False; for variables with multiple states, it is as defined in the .bif file.
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


def reduce_distribution(actual, believed):
    """
    Reduce a n-event probability distribution DataFrame to a (n-1)-event one.

    :param df4: DataFrame, with n event columns and a probability column.
    :param df3: DataFrame, with n-1 event columns and a probability column.
    :return: DataFrame, reduced probability distribution matching the (n-1)-event structure of believed.
    """
    # Select only the first 3 columns from df3 as keys
    believed_keys = believed.iloc[:, :-1]
    events = believed_keys.columns.tolist()

    # Group df4 by the first three columns (ignoring the fourth event) and sum probabilities
    actual_grouped = actual.groupby(events).agg({actual.columns[len(events)+1]: 'sum'}).reset_index()

    # Merge df4_grouped with df3 to keep only matching 3-event combinations
    df_reduced = pd.merge(believed_keys, actual_grouped, on=events, how='left')

    return df_reduced


def coherence(model,style="shogenji"):
    # style: "shogenji" (1999; keynes, 1923) or "og" (olsson, 2001; glass, 2001)
    # shogenji style
    nodes = list(model["model"].nodes)
    inference = bn.inference.fit(model, variables=nodes, evidence={}, verbose=0)
    inference_Df = bn.query2df(inference,verbose=0)
    max_p = inference_Df['p'].max()
    max_probability_rows = inference_Df[inference_Df['p'] == max_p]
    if len(max_probability_rows)>1:
       max_probability_rows = max_probability_rows.sample(1)

        #if there are multiple equiprobable and maximum probability states, go for a random one (this is a simplification, but it shouldn't have too much of an impact on the results, I hope)

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
    if len(actual_df.columns) != len(believed_df.columns):
        actual_df = reduce_distribution(actual_df, believed_df)
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