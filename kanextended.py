import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# advanced operation selector with mlp
class AdvancedOperationSelector(nn.Module):
    def __init__(self, hidden_dim):
        super(AdvancedOperationSelector, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# kan model with operation selection
class KANExtended(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_operations=2):
        super(KANExtended, self).__init__()
        self.num_operations = num_operations
        
        self.shared_psi_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(output_dim)])
        self.phi_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(output_dim * num_operations)])
        self.operation_selectors = nn.ModuleList([AdvancedOperationSelector(hidden_dim) for _ in range(output_dim * num_operations)])
        
        # final layer to reduce output to scalar
        self.final_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        shared_output = torch.relu(self.shared_psi_layer(x))
        sum_outputs = [torch.relu(output_layer(shared_output)) for output_layer in self.output_layers]
        
        final_output = 0
        
        for i, (phi_layer, selector) in enumerate(zip(self.phi_layers, self.operation_selectors)):
            phi_output = phi_layer(sum_outputs[i % len(sum_outputs)])
            operation_weight = selector(phi_output)
            product_term = operation_weight * phi_output
            sum_term = (1 - operation_weight) * phi_output
            
            if i % self.num_operations == 0:
                combined_output = product_term
            else:
                combined_output *= product_term
                
            combined_output += sum_term
            
            if i % self.num_operations == self.num_operations - 1:
                final_output += combined_output
        
        final_output = self.final_layer(final_output)
        return final_output

# custom loss function with regularization
def custom_loss(predicted, true, model, lambda_reg=0.01):
    mse_loss = nn.MSELoss()(predicted, true)
    reg_loss = sum(torch.sum(torch.abs(param)) for selector in model.operation_selectors for param in selector.parameters())
    return mse_loss + lambda_reg * reg_loss

# training function
def train_model(model, X, Y, epochs=1000, learning_rate=0.01, lambda_reg=0.01):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = custom_loss(outputs, Y, model, lambda_reg=lambda_reg)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'epoch [{epoch+1}/{epochs}], loss: {loss.item():.4f}')

# visualize all output elements of each layer with improved clarity
def visualize_layer_outputs_all_elements(model, sample_input):
    # pass the sample input through the shared_psi_layer to get hidden state
    hidden_state = torch.relu(model.shared_psi_layer(sample_input))
    
    model.eval()
    with torch.no_grad():
        plt.figure(figsize=(12, 8))
        
        # Track the x-position for each bar in the graph
        x_pos = 0
        bar_width = 0.8  # width of the bars
        
        for i, output_layer in enumerate(model.output_layers):
            # Get the output from each layer
            layer_output = torch.relu(output_layer(hidden_state))
            
            # Visualize all elements in the output (assuming the output has 10 elements)
            for j, output_value in enumerate(layer_output.squeeze().tolist()):
                # Plot the bar for this element
                plt.bar(x_pos, output_value, width=bar_width, color='b')
                x_pos += 1
        
        # Label X-axis and Y-axis more clearly
        plt.xlabel("Neurons in Each Layer (e.g., Layer 1 Neuron 1, Layer 1 Neuron 2, ...)", fontsize=12)
        plt.ylabel("Neuron Activation (Output) Value", fontsize=12)
        plt.title("Layer Outputs (Activation Values) for Input", fontsize=14)
        
        plt.show()





# main function to test specific multiplication of 2 * 3 = 6
def main():
    # input is [2, 3] and expected output is 6 (2 * 3)
    X = torch.tensor([[2.0, 3.0]])
    Y = torch.tensor([[6.0]])

    # create and train the model
    model = KANExtended(input_dim=2, output_dim=1, hidden_dim=10)
    train_model(model, X, Y, epochs=1000, learning_rate=0.01)
    
    # test the model with the same input to see if it predicts 6
    model.eval()
    with torch.no_grad():
        output = model(X)
        print(f"input: {X.numpy()}, predicted output: {output.item()}, expected output: 6")
    
    # visualize the operation selection for this specific test
    #visualize_layer_outputs_all_elements(model, X)

# run main
if __name__ == "__main__":
    main()
