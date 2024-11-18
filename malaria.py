#!/usr/bin/env python3
import argparse
import os
import pickle
import warnings

import networkx as nx
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from algorithms import randomlc, bayesopt
from depths import get_depths_from_bfs_layers

from gpytorch.utils.warnings import NumericalWarning

# ignore small noise warnings - NumericalWarning
warnings.filterwarnings("ignore", category=NumericalWarning)


def main():
    """
    Constraints
    b_exp - exponential of batch size; batch = (|depth|+1)^b_exp, where depth is the length of the longest
        file_name from the precursor to any of its evaluable postcursors. for b_exp=0, only one item is produced at a time
    c_base - Constant cost of purchasable (base) items
    c_synth - Constant synthesis cost of non-purchasable (non-base) items
    """

    # Define available files and algorithms
    files = [
        'master_16087_graph_processed',
        'subgraph_1000_seed0',
        'subgraph_5000_seed1',
        'subgraph_10000_seed2',
    ]
    algorithms = ['rs', 'lc5', 'lc20', 'bo', 'bopu']

    # Set up the argument parser
    parser = argparse.ArgumentParser(description='GECCO algorithm parameters')
    parser.add_argument('file', choices=files, help='File to process')
    parser.add_argument('algo', choices=algorithms, help='Algorithm to use')

    # Optional arguments with defaults
    parser.add_argument('--b_exp', type=float, default=2.0, help='Batch exponential')
    parser.add_argument('--c_base', type=int, default=1, help='Purchase cost of base items')
    parser.add_argument('--c_synth', type=int, default=1, help='Synthesis cost for precursors')
    parser.add_argument('--seed', type=int, default=123, help='Seed for random number generation')
    parser.add_argument('--output', type=str, help='Output directory')

    # Parse the arguments
    args = parser.parse_args()

    # create problem instance folder if not exists
    os.makedirs(args.output, exist_ok=True)
    file_name = f"{args.file}_b_exp_{args.b_exp}_c_base_{args.c_base}_c_synth_{args.c_synth}_{args.algo}_seed{args.seed}"
    print(file_name)

    with open(f'input/malaria/{args.file}.pkl', 'rb') as fn:
        dag = pickle.load(fn)

    # load smiles dataset
    with open('input/malaria/mal_smiles.txt', 'r') as data_f:
        raw_f_contents = data_f.read()
    target_smiles_all = raw_f_contents.split()

    # get target smiles from graph
    target_smiles = [x for x in dag.nodes() if dag.in_degree(x) == 0]
    # assert that all target smiles are in the dataset
    assert all([x in target_smiles_all for x in target_smiles])

    """ GRAPH ATTRIBUTES """
    dag.graph['target_smiles'] = target_smiles
    intermediate_topological_smiles = [mol for mol in list(nx.topological_sort(dag.reverse()))
                                       if dag.out_degree(mol) > 0]

    mol_idx = {mol: i for i, mol in enumerate(intermediate_topological_smiles)}

    dag.graph['intermediate_topological_smiles'] = intermediate_topological_smiles
    dag.graph['mol_idx'] = mol_idx

    """ Nodes ATTRIBUTES """
    # Compute shortest path lengths (depths) from multiple sources
    depths = get_depths_from_bfs_layers(dag, target_smiles)
    nx.set_node_attributes(dag, depths, 'depth')
    # compute batch constraint with exponential B_EXP
    nx.set_node_attributes(dag,
                           {mol: int((1 + dag.nodes[mol]['depth']) ** args.b_exp) for mol in dag.nodes()},
                           'batch')

    # Map fitness values to target nodes
    with open('input/malaria/mal_ec50_y.txt', 'r') as data_f:
        raw_f_contents = data_f.read()
    target_fit = raw_f_contents.split()

    # Turn minimization problem into maximization problem
    nx.set_node_attributes(dag, {mol: -float(fit) for mol, fit in zip(target_smiles_all, target_fit)}, 'fit')

    # Map Morgan fingerprint to target nodes
    morgan_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    nx.set_node_attributes(dag,
                           {mol: torch.tensor(list(
                               morgan_fp_gen.GetFingerprint(Chem.MolFromSmiles(mol))
                           ), dtype=torch.int8) for mol in target_smiles},
                           'fp')

    print('\nData Inputed')

    # Run algorithm
    if args.algo == 'rs':
        x_smiles, (cost, fit) = randomlc(dag=dag, num_evaluations=1000, c_base=args.c_base, c_synth=args.c_synth, seed=args.seed)
    elif args.algo in ['lc5', 'lc20']:
        x_smiles, (cost, fit) = randomlc(dag=dag, num_evaluations=1000, sample=int(args.algo[2:]),
                                    c_base=args.c_base, c_synth=args.c_synth, seed=args.seed)
    elif args.algo == 'bo':
        x_smiles, (cost, fit) = bayesopt(dag=dag, num_evaluations=300,
                                         c_base=args.c_base, c_synth=args.c_synth, seed=args.seed)
    elif args.algo == 'bopu':
        x_smiles, (cost, fit) = bayesopt(dag=dag, num_evaluations=300, per_cost=True,
                                         c_base=args.c_base, c_synth=args.c_synth, seed=args.seed)
    else:
        raise ValueError(f'Algorithm {args.algo} not recognized')

    results = { # Save results to a dictionary
        'x_smiles': x_smiles,
        'cost': cost,
        'fit': fit,
        'problem': args.file,
        'algo': args.algo,
        'b_exp': args.b_exp,
        'c_base': args.c_base,
        'c_synth': args.c_synth,
        'seed': args.seed,
    }

    with open(f"{args.output}/{file_name}.pkl", 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
