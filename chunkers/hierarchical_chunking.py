import pandas as pd
import torch
from nltk import PunktSentenceTokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer

from constants import MODEL_NAME, PRODUCT_DF_FILEPATH
from utils.embedding_utils import add_sos_and_bos, aggregate_embeddings


class HierarchicalChunker():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agg_method = 'mean'
        self.sentence_tokenizer = PunktSentenceTokenizer()
        self.model_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()

    def hierarchical_chunking(self, combined_review):
        print(combined_review)
        sentences_in_subreviews = [self.sentence_tokenizer.tokenize(sentence.strip()) for sentence in
                                   combined_review.split("||")]
        subreviews_embedded = list(map(self.embed_each_review, sentences_in_subreviews))

        subreview_token_count = [len(subreview) for subreview in subreviews_embedded]
        num_subreviews = len(subreviews_embedded)
        print(f'number of subreviews: {num_subreviews}')
        for i in range(num_subreviews):
            print(f'no. tokens in no. {i + 1} subreview: {subreview_token_count[i]}')
        print(f'all subreviews will be padded to {max(subreview_token_count)} tokens')
        padded_all_reviews = pad_sequence(subreviews_embedded,
                                          batch_first=True)  # get shape: (num subreviews, highest token len of subreviews, model embedding dim)

        return padded_all_reviews.mean(dim=0)

    def embed_each_review(self, subreview):
        tokenized_subreview = self.model_tokenizer(subreview, padding=True, truncation=False, return_tensors="pt")

        input_ids_full = tokenized_subreview['input_ids'].to(self.device)
        attn_mask_full = tokenized_subreview['attention_mask'].to(self.device)

        input_id_sentences = list(input_ids_full)
        mask_sentences = list(attn_mask_full)

        for i in range(len(input_id_sentences)):
            input_id_sentences[i], mask_sentences[i] = add_sos_and_bos(self, input_id_sentences[i], mask_sentences[i])

        input_ids = torch.stack(input_id_sentences).long()
        mask = torch.stack(mask_sentences)
        subreview_embeddings = aggregate_embeddings(self, input_ids, mask, self.agg_method)

        return subreview_embeddings

if __name__ == "__main__":
    torch.cuda.empty_cache()
    chunker = HierarchicalChunker()
    product_df = pd.read_csv(PRODUCT_DF_FILEPATH)

    sample_product_df = product_df.iloc[0:5].copy()
    sample_product_df.loc[:, 'hierar_chunk'] = sample_product_df['reviews'].apply(lambda review: chunker.hierarchical_chunking(review))
    print(f'shape of embedding for first entry: {sample_product_df.hierar_chunk.iloc[0].shape}')  # unique for each row - will be no. tokens x model embedding dim
    sample_product_df.head()