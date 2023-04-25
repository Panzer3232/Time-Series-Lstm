import torch
import torch.nn as nn
import torch.optim as optim
import data_load as dl
import model as m
import visualize as v

SimpleLSTM = m.SimpleLSTM

# Step 4: Train the neural network
def train_model(model, criterion, optimizer, train_inputs, train_targets, epochs=100):
    print("Training the model...")
    for epoch in range(epochs):
        inputs = torch.tensor(train_inputs, dtype=torch.float32)
        targets = torch.tensor(train_targets, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

input_size = 12
hidden_size = 32
output_size = 12
learning_rate = 0.01

model = SimpleLSTM(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_model(model, criterion, optimizer, dl.train_inputs, dl.train_targets)

# Step 5: Test the neural network and plot the results
def test_model(model, test_inputs):
    print("Testing the model...")
    inputs = torch.tensor(test_inputs, dtype=torch.float32)
    outputs = model(inputs).detach().numpy()
    return outputs

predictions = test_model(model, dl.test_inputs)
actual_values = dl.test_targets

v.visualize_results(actual_values, predictions)