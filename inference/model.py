import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size: int = 128, input_size: int = 1024, dropout_p: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs: torch.Tensor):
        embedding = self.embedding(inputs)
        output, (hidden, cell) = self.lstm(self.dropout(embedding))
        return output, (hidden, cell), embedding


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query: torch.Tensor, keys: torch.Tensor):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        weights = F.softmax(scores.permute(0, 2, 1), dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, dropout_p: float = 0.7):
        super().__init__()
        self.linear = nn.Linear(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.lstm = nn.LSTM(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.output_size = output_size
        self.MAX_LENGTH = 26

    def forward(self, encoder_outputs: torch.Tensor, encoder_hidden):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.zeros(
            batch_size, self.output_size, device=encoder_outputs.device
        )
        decoder_hidden = encoder_hidden
        outputs = []
        attentions = []

        for _ in range(self.MAX_LENGTH):
            output, decoder_hidden, attention = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            outputs.append(output)
            attentions.append(attention)
            decoder_input = F.softmax(output.squeeze(1), dim=-1)

        return torch.cat(outputs, dim=1), torch.cat(attentions, dim=1)

    def forward_step(self, inputs: torch.Tensor, hidden, encoder_outputs: torch.Tensor):
        hidden_state, cell_state = hidden
        embedded = self.linear(inputs).unsqueeze(1)
        query = hidden_state[-1].unsqueeze(1)
        context, attention = self.attention(query, encoder_outputs)
        output, (hidden_out, cell_out) = self.lstm(
            torch.cat((embedded, context), dim=2), (hidden_state, cell_state)
        )
        return self.out(output), (hidden_out, cell_out), attention


class Seq2Seq(nn.Module):
    def __init__(self, encoder: EncoderRNN, decoder: AttnDecoderRNN):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs: torch.Tensor):
        encoder_output, encoder_hidden, embedding = self.encoder(inputs)
        output, attention = self.decoder(encoder_output, encoder_hidden)
        return output, attention, embedding
