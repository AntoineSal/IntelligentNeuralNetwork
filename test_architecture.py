import torch
import torch.nn as nn
import torch.optim as optim
from src.intelligent_network import IntelligentNetwork

def test_forward_pass():
    print("=== Test Forward Pass ===")
    # Paramètres
    batch_size = 4
    seq_len = 10
    input_size = 5
    output_size = 2
    
    # Instanciation
    model = IntelligentNetwork(
        input_size=input_size,
        output_size=output_size,
        num_input_neurons=3,
        num_transmission_neurons=5,
        num_action_neurons=2,
        neuron_hidden_dim=16,
        key_query_dim=8,
        signal_dim=8
    )
    
    # Données factices (Batch, Time, Input)
    input_data = torch.randn(batch_size, seq_len, input_size)
    
    # Forward
    output = model(input_data)
    
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")
    
    expected_shape = (batch_size, seq_len, output_size)
    assert output.shape == expected_shape, f"Erreur de dimension: {output.shape} != {expected_shape}"
    print("Forward pass réussi !\n")

def test_training_step():
    print("=== Test Training Step (Backward) ===")
    batch_size = 4
    seq_len = 5
    input_size = 5
    output_size = 2
    
    model = IntelligentNetwork(
        input_size=input_size,
        output_size=output_size,
        num_input_neurons=2,
        num_transmission_neurons=3,
        num_action_neurons=2
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    input_data = torch.randn(batch_size, seq_len, input_size)
    target_data = torch.randn(batch_size, seq_len, output_size)
    
    # Reset gradients
    optimizer.zero_grad()
    
    # Forward
    output = model(input_data)
    
    # Loss
    loss = criterion(output, target_data)
    print(f"Initial Loss: {loss.item()}")
    
    # Backward
    loss.backward()
    
    # Check gradients
    has_grads = True
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"Warning: Pas de gradient pour {name}")
            has_grads = False
    
    if has_grads:
        print("Tous les paramètres ont reçu des gradients.")
        
    # Step
    optimizer.step()
    print("Training step réussi !\n")

if __name__ == "__main__":
    test_forward_pass()
    test_training_step()

