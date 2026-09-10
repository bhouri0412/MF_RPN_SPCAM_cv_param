## Multi-fidelity climate model parameterization for better generalization and extrapolation

Code and data accompanying the manuscript titled "Multi-fidelity climate model parameterization for better generalization", authored by Blanka Balogh, Mohamed Aziz Bhouri, Liran Peng, Michael S. Pritchard, Rose Yu and Pierre Gentine.

## Abstract
Machine-learning-based parameterizations (ie., representation of sub-grid processes) of global climate models have recently been proposed as a powerful alternative to physical, but empirical, representations, offering a lower computational cost and higher accuracy. Yet, those approaches still suffer from a lack of generalization and extrapolation beyond the high-fidelity training data, which is however critical to projecting climate change. 
Here we show that a multi-fidelity approach can provide the best of both worlds: it generalizes to a warmer climate for which high fidelity data are not available, guided by abundant low-fidelity data from an even warmer climate, while retaining the accuracy of high-fidelity data, by learning a nonlinear mapping from low- to high-fidelity. 

In an application to climate modeling, the multi-fidelity framework yields more accurate climate projections without requiring major increase in computational resources, while providing uncertainty estimates that increase coherently with prediction error. Even though the multi-fidelity model was only tested offline, our approach paves the way for the use of machine-learning based methods that can optimally leverage historical observations or high-fidelity simulations and extrapolate to unseen regimes such as warmer climates.
