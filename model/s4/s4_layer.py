from .s4 import S4
import torch
import torch.nn as nn


class S4Layer(nn.Module):

    def __init__(
        self, 
        d_input,
        d_s4_state=64,
        d_output=10, 
        d_model=256, 
        dropout=0,
        max_seq_len=1024,
        conv_validation=True
    ):
        super().__init__()

        self.conv_validation = conv_validation
        self.d_s4_state = d_s4_state

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        self.s4_layer = S4(
                    d_model=d_model, # H
                    d_state=d_s4_state, # N
                    l_max=max_seq_len, # L (L_Max)
                    dropout=0.0, 
                    mode='diag', 
                    measure='diag-lin',
                    disc='zoh', real_type='exp',
                    transposed=False, 
                    activation='gelu')
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout2d(dropout)

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def s4_rnn_style_forward(self, layer, sample, hidden_state=None):
        assert not self.training, f"RNN Style Forward used only in Eval mode!"

        # Setup: required for S4 modules in RNN Style Forward pass
        for module in layer.modules():
            if hasattr(module, 'setup_step'): module.setup_step()
            
        # Initialize
        batch_size, T, d_model = sample.shape

        if hidden_state is None:
            hidden_state = torch.zeros((batch_size, d_model, self.d_s4_state//2))

        out = torch.empty(batch_size, 0, d_model)

        # Do forward pass
        for t in range(0, T):
            out_step, hidden_state = layer.step(u=sample[:,t,:], state=hidden_state)
            out = torch.cat((out, out_step.unsqueeze(1)), dim=1)

        return out, hidden_state

    def s4_layer_forward(self, layer, sample, hidden_state=None, input_lengths=None):
        if self.training or self.conv_validation:
            # Do a Convolution-Style Forward Pass
            out, hidden_state =  layer.forward(u=sample, state=hidden_state, lengths=input_lengths)
        else:
            # Else, do a RNN Style Forward Pass
            assert input_lengths is None; f"For RNN-Style Forward Pass, we support BatchSize=1 only. Hence input_lengths should be None"
            out, hidden_state = self.s4_rnn_style_forward(layer=layer, sample=sample, hidden_state=hidden_state)
        return out, hidden_state


    def forward(self, x, hidden_state=None, input_lengths=None):
        """
        Input x: (B, seq_len, d_input)
        Input hidden_state: (B, d_model, d_s4_state)
        input_lengths: (B,)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        
        z = x

        z, _ = self.s4_layer_forward(layer=self.s4_layer, sample=z, hidden_state=hidden_state, input_lengths=input_lengths)

        # Dropout on the output of the S4 block
        z = self.dropout(z.transpose(-1, -2)).transpose(-1, -2)

        # Residual connection
        x = z + x

        # Postnorm
        x = self.layer_norm(x)

        # Decode the outputs
        x = self.decoder(x)  # (B, L, d_model) -> (B, L, d_output)
        return x



if __name__ == '__main__':
    batch_size = 3 # B
    seq_length = 10 # L

    d_model = 256 # H

    d_state = 64 # N
    max_seq_len = 50

    dropout = 0

    input_dim = 10
    output_dim = 10


    model = S4Layer(d_input=input_dim,
        d_s4_state=d_state,
        d_output=output_dim, 
        d_model=d_model, 
        dropout=dropout,
        max_seq_len=max_seq_len,
    )

    # print(model)

    """
    Generate Random Sample, and define initial state
    """
    mean = 0
    std = 0.5
    sample = torch.randn(batch_size, seq_length, input_dim) * std + mean
    initial_state = torch.zeros((batch_size, d_model, d_state//2))

    input_lengths = torch.Tensor([5, 10, 7])

    """
    Do Output using Conv-FFT (used during training)
    """
    conv_out = model(x=sample, hidden_state=initial_state, input_lengths=input_lengths)

    """
    Do Output using RNN style forward pass (used during Eval)
    """
    model.eval()
    rnn_out = model(x=sample, hidden_state=initial_state)

    # """
    # Compare Conv-Style and RNN-Style Forward pass Outputs (and final hidden state)
    # """
    # assert torch.allclose(rnn_out, conv_out, rtol=1e-3, atol=1e-3), f"S4 Outputs for Conv Style fwd pass, does not match RNN style forward pass"


