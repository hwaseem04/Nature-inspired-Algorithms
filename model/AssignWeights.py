def assign_weights(best_chromosome, model):
    input_nodes = 11
    hidden_nodes_1 = 8
    hidden_nodes_2 = 4
    output_nodes = 1
    
    weights = best_chromosome

    # reshaping for hidden layer 1's weights and bias
    hidden_layer_1_weights = weights[:input_nodes * hidden_nodes_1].reshape(hidden_nodes_1, input_nodes)
    used = input_nodes * hidden_nodes_1
    hidden_layer_1_bias = weights[used : used + hidden_nodes_1].reshape(hidden_nodes_1, 1)
    used += hidden_nodes_1 

    # reshaping for hidden layer 2's weights and bias   
    weights = weights[used:].copy()
    hidden_layer_2_weights = weights[:hidden_nodes_1 * hidden_nodes_2].reshape(hidden_nodes_2, hidden_nodes_1)
    used = hidden_nodes_1 * hidden_nodes_2
    hidden_layer_2_bias = weights[used : used + hidden_nodes_2].reshape(hidden_nodes_2, 1)
    used += hidden_nodes_2 

    # reshaping for output layer's weights and bias
    weights = weights[used:].copy()
    output_layer_weights = weights[: hidden_nodes_2 * output_nodes].reshape(output_nodes, hidden_nodes_2)
    used = hidden_nodes_2 * output_nodes
    output_layer_bias = weights[used : ].reshape(1,1)
    
    model.W1 = hidden_layer_1_weights
    model.B1 = hidden_layer_1_bias

    model.W2 = hidden_layer_2_weights
    model.B2 = hidden_layer_2_bias

    model.W3 = output_layer_weights
    model.B3 = output_layer_bias
    
    # print(model.W1.shape, model.W2.shape, model.W3.shape)
    return model
