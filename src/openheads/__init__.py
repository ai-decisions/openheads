"""openheads: the training method behind an on-chain screening backbone.

Model architectures, behavioural feature engineering, label tooling and
training utilities for node classification on blockchain transaction graphs.
The production serving stack is NOT part of this package (see the README's
Boundary section); `inference.GNNScorer` is a local scoring utility for
checkpoints this repository trains, not that serving stack.

No re-exports on purpose: import from the submodule that owns the name.
"""

__version__ = "0.1.0"
