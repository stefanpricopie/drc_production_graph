import pickle
from typing import Union, List, Optional

import networkx as nx
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from acqf import optimize_acqf_discrete
from tanimoto import TanimotoGP


def calculate_cost(storage: torch.Tensor, dag: nx.DiGraph, x: Optional[Union[str, List[str]]] = None, c_base: float = 1, c_synth: float = 1) -> torch.Tensor:
    if x is None:
        intermediate_topological_smiles = dag.graph['intermediate_topological_smiles']
        mol_idx = dag.graph['mol_idx']
        x = dag.graph['target_smiles']
    else:
        if isinstance(x, str):
            x = [x]

        successors = set()
        for x_ in x:
            successors.update(nx.descendants(dag, x_))

        subgraph_nodes = set(x).union(successors)

        dag_intermediate_idx = torch.tensor([dag.graph['mol_idx'][mol] for mol in subgraph_nodes
                                             if dag.out_degree(mol) > 0],
                                            dtype=torch.int).sort(dim=0).values

        intermediate_topological_smiles = [dag.graph['intermediate_topological_smiles'][i] for i in dag_intermediate_idx]

        storage = storage[dag_intermediate_idx]

        mol_idx = {mol: i for i, mol in enumerate(intermediate_topological_smiles)}

    total_cost_torch = torch.full((len(intermediate_topological_smiles),), c_synth, dtype=torch.int)
    resource_reuse_torch = torch.zeros(len(intermediate_topological_smiles), len(intermediate_topological_smiles), dtype=torch.int)
    
    for i, node in enumerate(intermediate_topological_smiles):
        # get the quantity of the node
        q_node = dag.nodes[node]['batch']

        # 1. add base_successors cost
        # split successors into base/root nodes and non-base nodes
        intermediate_successors_idx = torch.tensor([mol_idx[s] for s in dag.successors(node) if dag.out_degree(s) > 0], dtype=torch.int)

        number_of_base_successors = len(list(dag.successors(node))) - len(intermediate_successors_idx)
    
        total_cost_torch[i] += number_of_base_successors * q_node * c_base
    
        # 2. store resource reuse of the node
        resource_reuse_torch[i, intermediate_successors_idx] = q_node
        overflow_idx_old = len(intermediate_topological_smiles)

        # 3. surplus from successors
        while torch.gt(resource_reuse_torch[i], storage).any():
            # select the last index where resource reuse is larger than storage
            overflow_idx = torch.where(torch.gt(resource_reuse_torch[i], storage))[0][-1]

            # raise error if overflow_idx is greater or equal to the previous overflow_idx
            if overflow_idx >= overflow_idx_old:
                raise ValueError("Overflow index is greater or equal to the previous overflow index")
            overflow_idx_old = overflow_idx

            # add the excess to the total cost
            total_cost_torch[i] += total_cost_torch[overflow_idx]

            # add resource_reuse of the overflow_mol to the resource_reuse_node
            resource_reuse_torch[i] += resource_reuse_torch[overflow_idx]

            # delete resource_reuse_node for the overflow_mol. NOTE - any molecule can only overflow once
            resource_reuse_torch[i, overflow_idx] = 0

    # only return the cost of the target nodes, not the entire graph
    return total_cost_torch[torch.tensor([mol_idx[s] for s in x], dtype=torch.int)]


def update_storage(storage: torch.Tensor, dag: nx.DiGraph, x: str, quantity: int = 1) -> None:
    # get index of x in storage
    x_idx = dag.graph['mol_idx'][x]
    if storage[x_idx] >= quantity:
        # consume quantity
        storage[x_idx] -= quantity
    else:
        # produce new batch of x
        storage[x_idx] += dag.nodes[x]['batch'] - quantity

        # update storage of successors
        for s in dag.successors(x):
            # only update non-base nodes
            if dag.out_degree(s) > 0:
                update_storage(storage=storage, dag=dag, x=s, quantity=dag.nodes[x]['batch'])


def randomlc(dag, sample: int = 1, num_evaluations: int = 1000,
             c_base: float = 1, c_synth: float = 1,
             seed: int = None):

    # fix seed
    torch.manual_seed(seed)

    # Initialize storage space for resources as a torch tensor
    storage = torch.zeros(len(dag.graph['intermediate_topological_smiles']), dtype=torch.int)

    # Get target smiles from graph
    target_smiles = dag.graph['target_smiles']

    # Create lists to collect outputs
    output_x = []
    output_c_x = []
    output_fit = []

    # number of sources
    total_targets = len(target_smiles)

    for _ in range(min(num_evaluations, total_targets)):
        if len(target_smiles) == 0:
            break

        # Generate random indices for selection using torch.randperm
        indices = torch.randperm(len(target_smiles))[:min(len(target_smiles), sample)]

        # Select candidates using the generated random indices
        cand_list = [target_smiles[i] for i in indices]

        if sample > 1:
            # Evaluate candidate costs
            cand_cost = calculate_cost(storage=storage, dag=dag, x=cand_list, c_base=c_base, c_synth=c_synth)

            # Get the minimum cost and the corresponding index
            c_x, min_idx = torch.min(cand_cost, dim=0)
            x = cand_list[min_idx]
        else:
            # Evaluate the cost of the only candidate
            x = cand_list[0]
            c_x = calculate_cost(storage=storage, dag=dag, x=x, c_base=c_base, c_synth=c_synth)[0]

        # Update storage with the selected x
        update_storage(storage=storage, dag=dag, x=x)

        # Remove the evaluated node from the unevaluated list
        target_smiles.remove(x)

        # Append results to output lists
        output_x.append(x)
        output_c_x.append(c_x)
        output_fit.append(dag.nodes[x]['fit'])

    # Convert numeric lists to torch tensors
    output_c_x = torch.tensor(output_c_x)
    output_fit = torch.tensor(output_fit)

    # Return output_x as a list of strings and other outputs as stacked tensors
    return output_x, torch.stack((output_c_x, output_fit))


def bayesopt(dag, num_evaluations: int = 100, per_cost: bool = False,
             c_base: float = 1, c_synth: float = 1,
             seed: int = None):
    """
    Perform Bayesian Optimization using fingerprints from a graph of SMILES molecules.

    :param dag: A graph where each node contains a 'smiles', 'fp' and 'fit' attribute.
    :param num_evaluations: Number of BO iterations to perform.
    :param per_cost: If True, divide the acquisition function by the cost of each candidate.
    :param c_base: Base cost for each base molecule.
    :param c_synth: Synthesis cost for each molecule.
    :param seed: Random seed for reproducibility.
    :return: train_x (fingerprints of the SMILES evaluated),
             train_y (objective values for those SMILES),
             train_smiles (the SMILES strings or node names evaluated).
    """
    # fix seed
    torch.manual_seed(seed)

    # Get the target smiles
    target_smiles = dag.graph['target_smiles']

    # All bit candidates
    all_candidates = [dag.nodes[smiles]['fp'] for smiles in target_smiles]  # all_candidates and target_smiles are aligned
    y_values = nx.get_node_attributes(dag, 'fit')

    # Candidates (x) are binary, but target values (y) are continuous, and x and y need to be of same type
    # Stack adds a new dimension on the 0th dimension
    all_candidates = torch.stack(all_candidates).type(torch.double)

    # Construct vector to smiles dictionary
    vector_to_smiles = {tuple(fp.tolist()): smiles for fp, smiles in zip(all_candidates, target_smiles)}

    # Randomly sample initial points from the candidate set
    n_initial_points = 5
    initial_indices = torch.randperm(all_candidates.size(0))[:n_initial_points]
    train_x = all_candidates[initial_indices]
    train_smiles = [target_smiles[i] for i in initial_indices]
    train_y = torch.tensor([y_values[smiles] for smiles in train_smiles], dtype=torch.double).unsqueeze(-1)

    # Initialize storage space for resources as a torch tensor
    storage = torch.zeros(len(dag.graph['intermediate_topological_smiles']), dtype=torch.int)
    output_cost = []
    for x in train_smiles:
        c_x = calculate_cost(storage=storage, dag=dag, x=x, c_base=c_base, c_synth=c_synth)[0]
        output_cost.append(c_x)
        update_storage(storage=storage, dag=dag, x=x)
    output_cost = torch.tensor(output_cost)

    for _ in range(num_evaluations):
        # Fit a GP model to the data
        std_y = standardize(train_y).squeeze(-1)

        gp = TanimotoGP(train_x, std_y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Define the acquisition function (Expected Improvement)
        acq_func = ExpectedImprovement(gp, best_f=std_y.max())

        # Optimize the acquisition function over the discrete set
        x_candidates, acq_values = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=all_candidates,
            # (Default: 2048) The maximum number of choices to evaluate in batch. A large limit can cause excessive
            # memory usage if the model has a large training set.
            max_batch_size=2048,
            # (Default: True) If True return unique choices, o/w choices may be repeated (only relevant if `q > 1`).
            unique=True,
        )

        if per_cost:
            # Calculate the cost of all candidates. x=None defaults to target_smiles
            cand_cost = calculate_cost(storage=storage, dag=dag, x=None, c_base=c_base, c_synth=c_synth)

            # Divide the acquisition function by the cost of each candidate
            # acq_values and cand_cost are aligned because all_candidates is aligned with target_smiles
            acq_values = acq_values / cand_cost

            # Get the best candidate
            best_idx = torch.argmax(acq_values)
            new_x = x_candidates[best_idx]

            # Identify the SMILES corresponding to new_x
            new_smiles = vector_to_smiles[tuple(new_x.squeeze(0).tolist())]

            # Get the cost of the best candidate
            c_x = cand_cost[best_idx]
        else:
            # Get the best candidate
            best_idx = torch.argmax(acq_values)
            new_x = x_candidates[best_idx]

            # Identify the SMILES corresponding to new_x
            new_smiles = vector_to_smiles[tuple(new_x.squeeze(0).tolist())]

            # Calculate the cost of the best candidate
            c_x = calculate_cost(storage=storage, dag=dag, x=new_smiles, c_base=c_base, c_synth=c_synth)[0]

        # Update storage with the selected x
        update_storage(storage=storage, dag=dag, x=new_smiles)

        # Get fit value from new_smiles
        new_y = torch.tensor([y_values[new_smiles]], dtype=torch.double).unsqueeze(-1)

        # Add the new data to the training set
        train_x = torch.cat([train_x, new_x], dim=0)
        train_y = torch.cat([train_y, new_y], dim=0)
        output_cost = torch.cat([output_cost, c_x.unsqueeze(-1)], dim=0)
        train_smiles.append(new_smiles)

    # Return the evaluated fingerprints, their objective values, and corresponding SMILES strings
    return train_smiles, torch.stack((output_cost, train_y.squeeze(-1)))

