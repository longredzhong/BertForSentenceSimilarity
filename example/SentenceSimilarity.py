#%%
import sys
import os
sys.path.append(
    (os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))))
#%%
import torch
from model.AlbertForSequenceClassification import AlbertForSequenceClassification
from utils.tokenizers import Tokenizer
if __name__ == "__main__":
    net = AlbertForSequenceClassification.from_pretrained(
        "/home/longred/BertForSentenceSimilarity/output/LCQMC")
    tokenizer = Tokenizer(
        "/home/longred/BertForSentenceSimilarity/prev_trained_model/albert_tiny_zh/vocab.txt")
    sentence_1 = "这腰带是什么牌子"
    sentence_2 = "护腰带什么牌子好"
    input_ids ,segment_ids = tokenizer.encode(sentence_1,sentence_2)
    out = net(input_ids=torch.tensor(input_ids).unsqueeze(0),
                token_type_ids= torch.tensor(segment_ids).unsqueeze(0))
    print(out)


# %%
