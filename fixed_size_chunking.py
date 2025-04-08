import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from constants import PRODUCT_DF_FILEPATH, MODEL_NAME, TOKEN_LENGTH_PER_CHUNK


class FixedSizeChunker():
    def __int__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()

    def fixed_size_chunking(self, review):
        tokenized = self.tokenizer(review, padding=False, truncation=False, return_tensors="pt")
        input_ids_full = tokenized['input_ids'].squeeze(0).to(self.device)
        attn_mask_full = tokenized['attention_mask'].squeeze(0).to(self.device)

        input_id_chunks = list(input_ids_full.split(TOKEN_LENGTH_PER_CHUNK - 2))
        mask_chunks = list(attn_mask_full.split(TOKEN_LENGTH_PER_CHUNK - 2))

        # for chunk in input_id_chunks:
        # print(len(chunk))

        for i in range(len(input_id_chunks)):
            input_id_chunks[i] = torch.cat(
                (torch.Tensor([101]).to(self.device), input_id_chunks[i], torch.Tensor([102]).to(self.device)))
            mask_chunks[i] = torch.cat(
                [torch.Tensor([1]).to(self.device), mask_chunks[i], torch.Tensor([1]).to(self.device)])

            req_pad_len = TOKEN_LENGTH_PER_CHUNK - input_id_chunks[i].shape[0]

            if req_pad_len > 0:
                input_id_chunks[i] = torch.nn.functional.pad(input_id_chunks[i], (0, req_pad_len),
                                                             value=self.tokenizer.pad_token_id)
                mask_chunks[i] = torch.nn.functional.pad(mask_chunks[i], (0, req_pad_len), value=0)

        return torch.stack(input_id_chunks).long(), torch.stack(mask_chunks)

    def aggregate_embeddings(self, input_id_chunks, attn_mask_chunks, method='mean'):
        output = self.model(input_id_chunks, attn_mask_chunks)
        if method == 'mean':
            return output.last_hidden_state.mean(dim=0)
            # check again
        if method == 'maxpool':
            return output.last_hidden_state.max(dim=0)

    def chunk_and_embed(self, review, method='mean'):
        input_chunks, mask = self.fixed_size_chunking(review)
        print(f'number of chunks: {input_chunks.shape[0]}')
        input_chunks = input_chunks.to(self.device)
        mask = mask.to(self.device)
        embeddings = self.aggregate_embeddings(input_chunks, mask, method)
        return embeddings


if __name__ == "__main__":
    chunker = FixedSizeChunker()
    product_df = pd.read_csv(PRODUCT_DF_FILEPATH)

    sample_product_df = product_df.iloc[0:10]
    sample_product_df['embedding_fixed_chunk'] = [chunker.chunk_and_embed(review) for review in sample_product_df['reviews']]
    print(f'shape of embedding: {sample_product_df.embedding_fixed_chunk.iloc[0].shape}')
    print(sample_product_df.head())
