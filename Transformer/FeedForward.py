from torch import nn


class FeedForwardNet(nn.Module):
    def __init__(self, input_size, d_ff):
        """
        :param input_size:
        :param d_ff: FeedForward dimension
        """
        super(FeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, input_size, bias=False)
        )

        self.Norm = nn.LayerNorm(input_size)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        # [batch_size, seq_len, d_model]
        return self.Norm(output + residual)
