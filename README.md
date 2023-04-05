# Nature-inspired-Algorithms
Training neural network weights using nature inspired algorithms instead of gradient descent and back propagation.

## Algorihtms 
- **Genetic Algorithm**
    - Initially implemented GA for a toy dataset for a regression problem, refer [notebook](GeneticAlgorithm/GeneticAlgorithm.ipynb).
    - Not very impressive results on toy dataset. It stuck to local optima as can be clearly seen in the graph in the notebook.
    - No much changes made except in the neural network structure definition and stored the code in `GA.py` file in [model folder](model) so that I can import it in the main jupyter file - [Neural Network](Neural Network.ipynb)

- **Particle Swarm optimization**
    - Initially implemented it as in [Ali Mirjalili's lecture](https://youtu.be/JhgDMAm-imI). Refer [notebook](ParticleSwarmOptimisation/PSO.ipynb) for better understanding.
    - After that made little modifications (for weight updation) and stored the code in `PSO.py` file in [model folder](model) so that I can import it in the main jupyter file - [Neural Network](Neural Network.ipynb)
    - **Worth to be noted**: More influence is given to cognitive factor(a.k.a personal best values) in order to enhance exploitation.

- **Ant Colony Optimization**
    - Similarly like for PSO, I implemented generic ACO by reproducing the lecture of [Ali Mirjalili](https://youtu.be/783ZtAF4j5g). Refer [notebook](AntColonyOptimisation/ACO.ipynb) for better understanding.
    - Quite a lot of modifications made to suit the weight optimization problem for neural networks and stored the code in `ACO.py` file in [model folder](model) so that I can import it in the main jupyter file - [Neural Network](Neural Network.ipynb)
    - **Worth to be noted**: As far as in my implementation, this is the poorest performing optimisation algorithm *in context of neural network weight optimization*. For results see below.

## Results

Algorithm | GA     | PSO    | ACO    | 
--------- | ------ | ------ | ------ |
**Accuracy**  | **93.84%** | 93.52% | 90.56% | 