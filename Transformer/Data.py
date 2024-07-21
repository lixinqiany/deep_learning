import torch
from torch.utils.data import Dataset, DataLoader

class TransformerDataset(Dataset):
    """
    @:param samples raw & non-Tensor sequences
    @:param src the vocabulary of source language
    @:param tgt the vocabulary of target language
    """
    def __init__(self, samples, src, tgt):
        super(TransformerDataset, self).__init__()
        self.enc_inputs, self.dec_inputs, self.dec_outputs = self._convert2tensor(
            samples,
            src,
            tgt
        )
        print("Training Data Loaded...")

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

    """
    @:param samples the converted data
    
    The raw input is python lists. 
    It needs to convert them into a pytorch acceptable format
    """
    def _convert2tensor(self, samples, src, tgt):
        enc_inputs, dec_inputs, dec_outputs = [], [], []
        for i in range(len(samples)):
            numeric_enc_in = [src[j] for j in samples[i][0].split()]
            numeric_dec_in = [tgt[j] for j in samples[i][1].split()]
            numeric_dec_out = [tgt[j] for j in samples[i][2].split()]

            enc_inputs.append(numeric_enc_in)
            dec_inputs.append(numeric_dec_in)
            dec_outputs.append(numeric_dec_out)

        return torch.LongTensor(enc_inputs), \
            torch.LongTensor(dec_inputs), \
            torch.LongTensor(dec_outputs)

