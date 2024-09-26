import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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

class KANExtended(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_operations=2, operation_mode='both'):
        super(KANExtended, self).__init__()
        self.num_operations = num_operations
        self.operation_mode = operation_mode
        
        self.shared_psi_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(output_dim)])
        self.phi_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(output_dim * num_operations)])
        self.operation_selectors = nn.ModuleList([AdvancedOperationSelector(hidden_dim) for _ in range(output_dim)])

        self.final_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        shared_output = torch.relu(self.shared_psi_layer(x))
        sum_outputs = [torch.relu(output_layer(shared_output)) for output_layer in self.output_layers]
        
        final_output = 0
        
        for i, sum_output in enumerate(sum_outputs):
            phi_values = [torch.relu(phi_layer(sum_output)) for phi_layer in self.phi_layers[i*self.num_operations:(i+1)*self.num_operations]]
            
            if self.operation_mode == 'both':
                operation_weight = self.operation_selectors[i](sum_output)
                combined_output = torch.stack(phi_values, dim=0).prod(dim=0) * operation_weight + torch.stack(phi_values, dim=0).sum(dim=0) * (1 - operation_weight)
            elif self.operation_mode == 'add':
                combined_output = torch.stack(phi_values, dim=0).sum(dim=0)
            elif self.operation_mode == 'mul':
                combined_output = torch.stack(phi_values, dim=0).prod(dim=0)

            final_output += combined_output
        
        final_output = self.final_layer(final_output)
        return final_output

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def custom_loss(predicted, true, model, lambda_reg=0.001):
    mse_loss = nn.MSELoss()(predicted, true)
    reg_loss = sum(torch.sum(torch.abs(param)) for selector in model.operation_selectors for param in selector.parameters())
    return mse_loss + lambda_reg * reg_loss

def train_model(model, X, Y, epochs=1000, learning_rate=0.001, lambda_reg=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = custom_loss(outputs, Y, model, lambda_reg=lambda_reg)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

def calculate_percentage_differences(model, X_test, Y_test, epsilon=1e-8):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).squeeze()
        expected = Y_test.squeeze()
        # Replace zeros in expected values with a small epsilon to avoid division by zero
        expected = torch.where(expected == 0, torch.tensor(epsilon, dtype=expected.dtype), expected)
        percentage_diff = torch.abs((predictions - expected) / expected) * 100
    return percentage_diff.tolist()

def compare_operation_modes(X_train, Y_train, X_test, Y_test, hidden_dim, epochs=1000, learning_rate=0.01):
    modes = ['add', 'mul', 'both']
    results = {}
    percentage_diff_arrays = {'add': [], 'mul': [], 'both': []}

    for mode in modes:
        model = KANExtended(input_dim=2, output_dim=1, hidden_dim=hidden_dim, operation_mode=mode)
        initialize_weights(model)
        train_model(model, X_train, Y_train, epochs=epochs, learning_rate=learning_rate)
        percentage_diff = calculate_percentage_differences(model, X_test, Y_test)
        average_percentage_diff = sum(percentage_diff) / len(percentage_diff)

        percentage_diff_arrays[mode] = percentage_diff  # Store percentage differences

        model.eval()
        with torch.no_grad():
            predictions = model(X_test).squeeze().tolist()
            inputs = X_test.tolist()
            expected_outputs = Y_test.squeeze().tolist()
        
        results[mode] = {
            "average_percentage_diff": average_percentage_diff,
            "percentage_differences": percentage_diff,
            "predictions": predictions,
            "inputs": inputs,
            "expected_outputs": expected_outputs
        }

        print(f"\nMode: {mode}")
        print("Inputs | Predictions | Expected Outputs | Percentage Difference")
        for inp, pred, exp, diff in zip(inputs, predictions, expected_outputs, percentage_diff):
            print(f"{inp} | {pred:.4f} | {exp:.4f} | {diff:.2f}%")

    print("\nSummary of Results:")
    for mode, result in results.items():
        avg_diff = result['average_percentage_diff']
        print(f"Operation Mode: {mode} - Average Percentage Difference: {avg_diff:.2f}%")

    return percentage_diff_arrays

def main():
    X_train = torch.rand(1000, 2) * 10  
    Y_train = (X_train[:, 0] * X_train[:, 1]).unsqueeze(1)

    X_test = torch.rand(20, 2) * 10
    Y_test = (X_test[:, 0] * X_test[:, 1]).unsqueeze(1)

    percentage_diff_arrays = compare_operation_modes(X_train, Y_train, X_test, Y_test, hidden_dim=10, epochs=500, learning_rate=0.001)

    
    for mode, diffs in percentage_diff_arrays.items():
        avg_diff = sum(diffs) / len(diffs)
        print(f"\nOverall Average Percentage Difference for {mode.capitalize()}: {avg_diff:.2f}%")

if __name__ == "__main__":
    main()
