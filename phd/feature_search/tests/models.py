import torch
from phd.feature_search.core.models.ensemble_models import MultipleLinear


def test_multiple_linear():
    # Create a small test case
    in_features = 2
    out_features = 3
    n_parallel = 2
    batch_size = 2
    
    # Create model
    model = MultipleLinear(in_features, out_features, n_parallel, bias=True)
    
    # Manually set weights and biases for testing
    # First parallel computation
    model.weight.data[0] = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])
    model.bias.data[0] = torch.tensor([0.1, 0.2, 0.3])
    
    # Second parallel computation
    model.weight.data[1] = torch.tensor([
        [2.0, 1.0],
        [4.0, 3.0],
        [6.0, 5.0]
    ])
    model.bias.data[1] = torch.tensor([0.4, 0.5, 0.6])
    
    # Create test input
    x = torch.tensor([
        [[1.0, 2.0],  # First parallel input for first batch
         [3.0, 4.0]], # Second parallel input for first batch
        [[5.0, 6.0],  # First parallel input for second batch
         [7.0, 8.0]]  # Second parallel input for second batch
    ])
    
    expected_output = torch.tensor([
        [ # batch 1
            [1*1 + 2*2 + 0.1, 1*3 + 2*4 + 0.2, 1*5 + 2*6 + 0.3], # parallel 1
            [3*2 + 4*1 + 0.4, 3*4 + 4*3 + 0.5, 3*6 + 4*5 + 0.6], # parallel 2
        ],
        [ # batch 2
            [5*1 + 6*2 + 0.1, 5*3 + 6*4 + 0.2, 5*5 + 6*6 + 0.3], # parallel 1
            [7*2 + 8*1 + 0.4, 7*4 + 8*3 + 0.5, 7*6 + 8*5 + 0.6], # parallel 2
        ]
    ])
    
    # Get actual output
    actual_output = model(x)
    
    print(actual_output)
    print(expected_output)
    
    # Check shapes
    assert actual_output.shape == (batch_size, n_parallel, out_features)
    
    # Check values
    torch.testing.assert_close(actual_output, expected_output)