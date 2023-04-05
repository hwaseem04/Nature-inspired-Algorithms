import numpy as np
import torch
import torch.nn as nn
from .AssignWeights import assign_weights

# Initialize genes(w) and population 
# neural network with 1 layer and 2 neurons
def initlize_population(pop_size=10, n_params=3):
    population = np.random.normal(0,0.1, size=(pop_size, n_params))
    return population

# Perform mutation with a mutation rate
def perform_mutation(chromosome, mutation_rate=0.5):
    for i in range(len(chromosome)):
        # To apply mutation rate
        if np.random.rand() < mutation_rate:
            # To randomly increase and decrease weight - normal distribution with mean=0, std=0.1
            chromosome[i] += np.random.normal(0,0.2)
    return np.array(chromosome)

# Perform cross over operation (often rarely)
# Here parent1 and parent2 are chromosomes
def perform_crossover(parent1, parent2):
    # 1-point cross over performed 
    pt0 = np.random.randint(0, len(parent1) - 1)
    child1 = np.concatenate((parent1[:pt0], parent2[pt0:]))
    child2 = np.concatenate((parent2[:pt0], parent1[pt0:]))
    
    # 2-point cross over
    pt1 = np.random.randint(0, len(parent1) - 1)
    pt2 = np.random.randint(0, len(parent1) - 1)
    if pt2 < pt1:
        pt1, pt2 = pt2, pt1
    
    child3 = np.concatenate((parent1[:pt1], parent2[pt1:pt2], parent1[pt2:]))
    return child1.reshape(1,-1), child2.reshape(1,-1), child3.reshape(1,-1)

# Regression Loss
# # Mean squared error --> lower is better
# def loss_fn(pred, target):
#     mse = np.mean((pred - target)**2)
#     return mse

# Classification Loss   
loss_fn = nn.BCEWithLogitsLoss()

# Fitness function - evaluate fitness of each chromosome in population 
def evaluate_fitness(model, population, X, y):
    all_loss = []
    for chromosome in population:
        model = assign_weights(chromosome, model)
        pred = model.predict(X)
        pred = torch.tensor(pred, dtype=torch.float32).T
        y = torch.tensor(y, dtype=torch.float32)
        loss = loss_fn(pred, y)
        all_loss.append(loss)
    return np.array(all_loss)    

# Roulette wheel based selection of population for next generation. Size=k    
def perform_selection(new_population, model, X, y, k=10):
    fitness = evaluate_fitness(model, new_population, X, y)
    fitness = fitness.reshape(-1)
    # sorting from lowest to highest
    sorted_fitness_index = np.argsort(fitness)
    fitness = fitness[sorted_fitness_index]
    new_population = new_population[sorted_fitness_index, :]
    
    # Roulette wheel selection
    prob = fitness/np.sum(fitness)
    cum_probs = np.cumsum(prob)
    selected_index = []
    for i in range(k):
        r = np.random.rand()
        index = np.where(cum_probs > r)[0][0]
        
        # Inverse index to give more priority to least values
        index = len(fitness) - 1 - index
        selected_index.append(index)
    
    selected_index = np.array(selected_index)
    
    # Selecting the corresponding population and loss for those index
    selected_population = new_population[selected_index, :]
    k_losses = fitness[selected_index]
    
    return selected_population, k_losses

def GeneticAlgorithm(model, pop, X, y, crossover_rate, mutation_rate, k):
    # Mutation
    for j in range(len(pop)):
        pop[j] = perform_mutation(pop[j], mutation_rate)
    # Crossover
    times = np.random.randint(1,5)
    if np.random.rand() < crossover_rate:
        for _ in range(times):
            p1, p2 = np.random.choice(len(pop), 2)
            child1, child2, child3 = perform_crossover(pop[p1], pop[p2])
            pop = np.concatenate([pop, child1, child2, child3], axis=0)
    # Selection
    pop, losses = perform_selection(pop, model, X, y, k)
    
    return pop, losses
    
    
