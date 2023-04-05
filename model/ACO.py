import numpy as np
import torch.nn as nn
import torch
from .AssignWeights import assign_weights

loss_fn_aco = nn.BCEWithLogitsLoss()

# Generate cost graph
def cost_graph(v=10, s=0.1):
    graph = np.diag((np.inf,)*v)
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            # When not a diagonal element, i.e no self loop
            if i != j:
                graph[i][j] = np.random.normal(0,s)
    return graph

# To enhance the diversity of cost graph
def random_fill(graph, s=0.2, f=0.4):
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            # When not a diagonal element, i.e no self loop
            if np.random.rand() < f:
                if i != j:
                    graph[i][j] = np.random.normal(0,s)
    return graph

# Generate Pheromone graph
def pheromone_graph(v=10):
    pheromone = np.ones((v,v)) #np.random.normal(0, 0.1, (v,v))
    return pheromone


# Selecting next vertex in a path(assigning weights)
def selectNextVectex(currVertex, graph, pheromone):
    alpha, beta = 1, 1 # Controlling the influence of pheromone and path length
    
    # Cost of moving to other vertex from current vertex
    possible_pathCost = np.power(1/graph[currVertex], beta)
    
    # Existing pheromone levels at other vertex from current vertex
    possible_pheromone = np.power(pheromone[currVertex], alpha)
    
    # Probability of moving to other vector from current vertex
    P = (possible_pathCost * possible_pheromone)/ np.dot(possible_pathCost, possible_pheromone)
    
    # Cumulative sum - for roulette wheel based path selection
    C = np.cumsum(P)
    
    # Roulette wheel selection
    r = np.random.rand()
    selected_vertex = np.where(C > r)[0][0] # Selecting the first vertex that satisfies the condition 
    
    return selected_vertex


def map_vertex_to_weight(graph, path):
    path_w = []
    prev = path[0]
    for vertex in path[1:]:
        path_w.append(graph[prev][vertex])
        prev = vertex
    return np.array(path_w)

# Evaluate cost 
def evaluate_cost(model, X, Y):
    pred = model.predict(X)
    pred = torch.tensor(pred, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32).T
    loss = loss_fn_aco(pred, Y)
    return loss.item()

# Finding all paths(setting weights) and their corresponding costs for each path
def generationOfAnts(graph, X, Y,model, pheromone, size=20, total_params=137):
    paths = []
    costs = []
    
    for i in range(size):
        r = np.random.randint(len(graph))
        path = [r] # vertices used for updating pheromone graph
        cost = 0
        prev = r
        
        for iteration in range(total_params):
            next_vertex = selectNextVectex(prev, graph, pheromone)
            while next_vertex == prev:
                next_vertex = selectNextVectex(prev, graph, pheromone)
                
            path.append(next_vertex) 
            prev = next_vertex
            
        path = np.array(path)
        path_w = map_vertex_to_weight(graph, path)
        model = assign_weights(path_w, model)
        cost = evaluate_cost(model, X, Y)
        
        paths.append(path)
        costs.append(cost)

    return np.array(paths), np.array(costs)



# Update pheromone level on paths after each generation 
## This implementation is not 'the most' accurate. Because pheromone level on paths ideally must be updated
## after each ant movement. But here it is updated only after a generation (10 ants = 1 generation)
def updatePheromones(paths, costs, pheromone, decay=0.4):
    # pheromone secreted by ant based on the quality of path's cost
    pheromone_secreted = 1/np.array(costs)
    
    # Vaporization in existing pheromone
    pheromone = (1-decay) * pheromone
    
    # Adding the newly secreated pheromone to path
    for i in range(len(paths)):
        # Ant i's path
        path = paths[i]
        for j in range(len(path)-1):
            # Adding ant i's pheromone to its entire path
            pheromone[path[j]][path[j+1]] += pheromone_secreted[i] 
    
    return pheromone
        
def ACO(graph, X, Y, model, ph, size):
    paths, costs = generationOfAnts(graph, X, Y,model, ph, size) # 20 ants = 20 paths
    ph = updatePheromones(paths, costs, ph)

    return ph, costs, paths