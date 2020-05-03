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
    # use albert
    net = AlbertForSequenceClassification.from_pretrained(
        "/home/longred/BertForSentenceSimilarity/output/LCQMC/albert/")
    tokenizer = Tokenizer(
        "/home/longred/BertForSentenceSimilarity/prev_trained_model/albert_tiny_zh/vocab.txt")
    sentence_1 = "这腰带是什么牌子"
    sentence_2 = "护腰带什么牌子好"
    input_ids ,segment_ids = tokenizer.encode(sentence_1,sentence_2)
    out = net(input_ids=torch.tensor(input_ids).unsqueeze(0),
                token_type_ids= torch.tensor(segment_ids).unsqueeze(0))
    print(out)

    # use ESIM
    net = AlbertForSequenceClassification.from_pretrained(
        "/home/longred/BertForSentenceSimilarity/output/LCQMC/ESIM/")
    tokenizer = Tokenizer(
        "/home/longred/BertForSentenceSimilarity/prev_trained_model/albert_tiny_zh/vocab.txt")
    sentence_1 = "这腰带是什么牌子"
    sentence_2 = "护腰带什么牌子好"
    input_1_ids = tokenizer.encode(sentence_1)[0]
    input_1_len = len(input_1_ids)
    input_2_ids = tokenizer.encode(sentence_2)[0]
    input_2_len = len(input_2_ids)
    out = net(input_1=input_1_ids,
              input_1_lengths=input_1_len,
              input_2=input_2_ids,
              input_2_lengths=input_2_len
             )
    print(out)
