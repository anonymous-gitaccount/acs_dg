# DeepGame For Graph Neural Network
## Description
An extension of [DeepGame](https://github.com/TrustAI/DeepGame) for Graph Neural Network.
The framework provides guarantees for upper bound and lower bound computation of the Maximum Safe Radius for a node in the graph against node attribute attack. CLEVER extended from [CLEVER](https://arxiv.org/abs/1801.10578) and [Pytorch CLEVER](https://github.com/joemathai/clever_robustness_pytorch) which is under MIT License.
## Environment Dependencies
torch-geometric           2.3.1

torch                     1.13.1

scikit-learn              1.0.2

python                    3.7

numpy                     1.21.6

[deeprobust](https://github.com/DSE-MSU/DeepRobust)                0.2.8

pandas                    1.3.5

scipy                     1.7.3
## Instruction
upper bound:
```
python main.py $dataset_name ub cooperative $node_index L2 $distance_budget $tau $number_of_hop
```
For example,
`python main.py Cora ub cooperative 1755 L2 40 1 2` is to compute the upper bound for the node 1755 in the Cora dataset, considering indirect attack of 2-hop neighbors and manipulation magnitude 1, and the distance bound is 40 in L2 distance.

lower bound:
```
python main.py $dataset_name lb cooperative $node_index L2 $distance_budget $tau $number_of_hop
```
## Branch
There are two branches of the codebase. The branch **main** deploys the deep_robust library for upper bound indirect attack feature reduction while the branch **non-reduced** does not reduce the dimensionality and use the entire search space. 

To switch between branches, use `git checkout $branch_name`
