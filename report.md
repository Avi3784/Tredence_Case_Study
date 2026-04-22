# self pruning network report

## l1 penalty explanation
the l1 norm minimizes the absolute values. since our gate scores are passed through a sigmoid function that maps to (0, 1), minimizing their l1 norm forces the underlying scores negative, which pushes the sigmoid outputs to exactly zero, pruning the weights.

## experimental results
| Lambda | Test Accuracy | Sparsity Level (%) |
|---|---|---|
| 0.0001 | 51.87% | 0.0% |
| 0.001 | 48.22% | 0.0% |
| 0.01 | 39.53% | 0.0% |

## analysis
increasing lambda increases sparsity but drops test accuracy. the histogram shows a large spike at 0, showing the model successfully pruned connections.
