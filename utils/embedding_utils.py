import torch


def add_sos_and_bos(self, sentence, mask):
    input_ids = torch.cat([torch.Tensor([101]).to(self.device), sentence, torch.Tensor([102]).to(self.device)])
    mask = torch.cat([torch.Tensor([1]).to(self.device), mask, torch.Tensor([1]).to(self.device)])
    return input_ids, mask


def aggregate_embeddings(self, input_id_chunks, attn_mask_chunks, method='mean'):
    output = self.model(input_id_chunks, attn_mask_chunks)
    if method == 'mean':
        return output.last_hidden_state.mean(dim=0)
        # check again
    if method == 'maxpool':
        return output.last_hidden_state.max(dim=0)