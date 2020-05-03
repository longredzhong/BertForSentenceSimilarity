#%%
import torch
from model.AlbertForSequenceClassification import AlbertForSequenceClassification
from dataloader.LCQMCDataloader import LCQMCDataLoader
from transformers.modeling_albert import AlbertConfig

from utils.train import train
from utils.evaluate import evaluate
if __name__ == "__main__":
    train_data_path = "/home/longred/BertForSentenceSimilarity/dataset/LCQMC/train.txt"
    dev_data_path = "/home/longred/BertForSentenceSimilarity/dataset/LCQMC/dev.txt"
    vocab_path = "/home/longred/BertForSentenceSimilarity/prev_trained_model/albert_tiny_zh/vocab.txt"

    train_data_loader = LCQMCDataLoader(train_data_path,vocab_path,batch_size=256,is_pair=True)
    dev_data_loader = LCQMCDataLoader(dev_data_path, vocab_path, batch_size=256, is_pair=True)

    config = AlbertConfig.from_pretrained(
        "/home/longred/BertForSentenceSimilarity/prev_trained_model/albert_tiny_zh/config.json")
    config.num_labels = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    net = AlbertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path="/home/longred/BertForSentenceSimilarity/prev_trained_model/albert_tiny_zh/pytorch_model.bin",
        config=config).to(device)
    # %%
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    # %%
    for i in range(10):
        trainepochloss = train(net, train_data_loader, optimizer, device)
        print(trainepochloss)
        evalepochloss, acc = evaluate(net, dev_data_loader, device)
        print(evalepochloss, acc)
    # %%
    net.save_pretrained("/home/longred/BertForSentenceSimilarity/output/LCQMC")
