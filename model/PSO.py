import torch
import numpy as np
import torch.nn as nn
from .AssignWeights import assign_weights

loss_fn_pso = nn.BCEWithLogitsLoss()

# initialize particles - position, velocity, personal best, global best, cost
def initialize(n=10, dim=137):
    particle_pos = np.random.normal(0, 0.1, (n,dim))
    particle_vel = np.zeros((n,dim))
    particle_best_pos = particle_pos.copy()
    particle_best_cost = np.full(n, np.inf)
    global_best_pos = None
    global_best_cost = np.zeros(dim)

    return particle_pos, particle_vel, particle_best_pos, particle_best_cost, global_best_cost

    
# Evaluate cost
def evaluate_cost(particle_pos, model, X, y):
    particle_cost = []
    
    for particle in particle_pos:
        model = assign_weights(particle, model)
        pred = model.predict(X)
        pred = torch.tensor(pred, dtype=torch.float32).T
        y = torch.tensor(y, dtype=torch.float32)
        loss = loss_fn_pso(pred, y)
#         print(loss)
        particle_cost.append(loss.item())
        
#         particle_cost = cost(particle_pos, cost_fn)
    return np.array(particle_cost)

# Update personal best
def updatePersonalBest(curr_particle_pos, curr_particle_cost, particle_best_pos, particle_best_cost):
    best_cost_mask = curr_particle_cost < particle_best_cost # Minimum is better
    particle_best_pos[best_cost_mask] = curr_particle_pos[best_cost_mask]
    particle_best_cost[best_cost_mask] = curr_particle_cost[best_cost_mask]
    return particle_best_pos, particle_best_cost

# Update global best
def updateGlobalBest(particle_best_pos, particle_best_cost):
    best_mask = np.argmin(particle_best_cost) # Minimum is better
    global_best_pos = particle_best_pos[best_mask]
    global_best_cost = particle_best_cost[best_mask]
    return global_best_pos, global_best_cost

# Update particle position
def updateParticle(curr_vel, curr_particle_pos, particle_best_pos, global_best_pos, limit):
    r1 = np.random.rand(1)
    r2 = np.random.rand(1)
    
    # Inertia factor
    w = 0.8
    
    # Cognitive and social factor
    c1, c2 = 3,1
    
    # Updating particle position with three components
    particle_vel = w * curr_vel + c1*r1*(particle_best_pos-curr_particle_pos) + c2*r2*(global_best_pos-curr_particle_pos)
    particle_pos = particle_vel + curr_particle_pos
    
    # If at all the position value explodes, clip it within the range
    particle_pos = np.clip(particle_pos, limit[0], limit[1])
    return particle_pos, particle_vel
    
def PCO(model, particle_pos, particle_vel, particle_best_pos, particle_best_cost, global_best_cost, X, Y, limit):

    particle_cost = evaluate_cost(particle_pos, model, X, Y)

    # Update personal best
    particle_best_pos, particle_best_cost = updatePersonalBest(
        particle_pos, particle_cost, particle_best_pos, particle_best_cost)

    # Update Global best
    global_best_pos, global_best_cost = updateGlobalBest(particle_best_pos, particle_best_cost)

    # Update particle position
    particle_pos, particle_vel = updateParticle(particle_vel, particle_pos, particle_best_pos, global_best_pos, limit)
        
    return particle_pos, particle_vel, particle_best_pos, particle_best_cost, global_best_cost
