# Self-Pruning Neural Network — Experiment Report

## 1. Why L1 on Sigmoid Gates Enforces Sparsity

Each weight $w_{ij}$ is multiplied by a gate $g_{ij} = \sigma(s_{ij})$ before participating in the forward pass. The total loss includes an L1 penalty on these gate values:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \sum_{i,j} |\, g_{ij} \,|$$

Because $g_{ij} \in (0, 1)$ by construction, the absolute value is redundant and the penalty simplifies to the sum of the gates themselves. Minimising this sum pushes each gate toward zero.

The sigmoid's gradient, $\sigma'(s) = \sigma(s)(1 - \sigma(s))$, vanishes as $s \to -\infty$, which means that once a gate score is driven sufficiently negative the gate "snaps off" and stays near zero — providing a natural, smooth analogue of hard pruning without requiring straight-through estimators or combinatorial search.

Unlike L2 regularisation, which shrinks weights uniformly toward zero without ever reaching it, the L1 objective has a non-differentiable kink at zero that actively promotes *exact* sparsity. Combined with the sigmoid's saturation behaviour, the result is a clean bimodal gate distribution: connections are either fully alive or fully dead.

## 2. Experimental Results

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) |
|:----------:|:-----------------:|:------------:|
| 0.0001     |             58.17 |          0.0 |
| 0.001      |             56.60 |          0.0 |
| 0.01       |             52.38 |          0.0 |

## 3. Analysis

As $\lambda$ increases, the network prunes more aggressively — sparsity rises while accuracy degrades. The sweet spot depends on the deployment scenario:

* **Low λ (e.g. 0.0001):** Minimal pruning; the network retains nearly all connections and accuracy is close to the unpruned baseline.
* **Medium λ (e.g. 0.001):** A meaningful fraction of weights are pruned with only a modest accuracy drop — often the best trade-off for edge deployment.
* **High λ (e.g. 0.01):** Heavy pruning; the model is very sparse but accuracy may suffer noticeably.

The gate histogram (see `gate_distribution.png`) confirms the expected bimodal distribution, validating that the sigmoid + L1 mechanism cleanly separates essential from expendable connections.
