## Multi-fidelity climate model parameterization for better generalization and extrapolation

Code and data accompanying the manuscript titled "Multi-fidelity climate model parameterization for better generalization and extrapolation", authored by Mohamed Aziz Bhouri, Liran Peng, Michael S. Pritchard, Rose Yu and Pierre Gentine.

## Abstract

Machine-learning-based parameterizations (i.e. subgrid representation of subgrid processes) of global climate models or turbulent simulations have recently been proposed as a powerful alternative to physical, but empirical, representations, offering a lower computational cost and higher accuracy. Yet, those approaches still suffer from a lack of generalization and extrapolation beyond the training data, which is however critical to projecting climate change or unobserved regimes of turbulence. Here we show that a multi-fidelity approach, which integrates datasets of different accuracy and abundance, can provide the best of both world: the capacity to extrapolate leveraging the  physically-based parameterization and higher accuracy using the machine-learning-based parameterizations. In an application to cliamte modeling, the multi-fidelity framework yields more accurate climate projections without requiring major increase in computational resources. Our multi-fidelity randomized prior networks (MF-RPNs) combine physical parameterization data as low-fidelity and storm-resolving historical run's data as high-fidelity. To extrapolate beyond the training data, the MF-RPNs are tested on high-fidelity warming scenarios, $+4K$, data. We show the MF-RPN's capacity to return much more skillful predictions compared to either low-fidelity simulations or high-fidelity trained only on one regime (historical data) while providing trustworthy uncertainty quantification across a wide range of regimes. Our approach paves the way for the use of machine-learning based methods that can optimally leverage historical observations or high-fidelity simulations and can extrapolate to unseen regimes such as climate change.

## Citation

    @article{Bhouri2023MFRPN,
    title = {Multi-fidelity climate model parameterization for better generalization and extrapolation},
    author = {Bhouri, Mohamed Aziz and Peng, Liran and Pritchard, Michael S. and Yu, Rose and Gentine, Pierre },
    journal = {arXiv preprint arXiv:},
    doi = {https://doi.org/},
    year = {2023},
    }
