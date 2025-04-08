from nltk import PunktSentenceTokenizer
import pandas as pd
import torch
import itertools
from transformers import AutoModel, AutoTokenizer, AutoConfig

from constants import MODEL_NAME, PRODUCT_DF_FILEPATH
from utils.embedding_utils import add_sos_and_bos, aggregate_embeddings


class SentenceChunker():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.sentence_tokenizer = PunktSentenceTokenizer()
        self.model_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()

    def sentence_chunking(self, combined_review):
        sentences = [self.sentence_tokenizer.tokenize(review.strip()) for review in combined_review.split("||")]
        sentences = list(itertools.chain(*sentences))
        print(f'number of sentences: {len(sentences)}')
        tokenized = self.model_tokenizer(sentences, padding=True, truncation=False, return_tensors="pt")

        input_ids_full = tokenized['input_ids'].squeeze(0).to(self.device)
        attn_mask_full = tokenized['attention_mask'].squeeze(0).to(self.device)

        input_id_sentences = list(input_ids_full)
        mask_sentences = list(attn_mask_full)

        for i in range(len(input_id_sentences)):
            input_id_sentences[i], mask_sentences[i] = add_sos_and_bos(self, input_id_sentences[i], mask_sentences[i])

        print(f'number of tokens in each sentence: {len(input_id_sentences[0])}')

        return torch.stack(input_id_sentences).long(), torch.stack(mask_sentences)

    def chunk_and_embed(self, review, method='mean'):
        input_chunks, mask = self.sentence_chunking(review)
        print(f'number of chunks for sentence chunking: {input_chunks.shape[0]}')
        input_chunks = input_chunks.to(self.device)
        mask = mask.to(self.device)
        embeddings = aggregate_embeddings(input_chunks, mask, method)
        return embeddings


if __name__ == "__main__":
    chunker = SentenceChunker()
    product_df = pd.read_csv(PRODUCT_DF_FILEPATH)

    sample_product_df = product_df.iloc[0:5]
    sample_product_df['sentence_chunk'] = sample_product_df['reviews'].apply(lambda review: chunker.chunk_and_embed(review))
