#%%
import torch
from model.AlbertForSequenceClassification import AlbertForSequenceClassification
from dataloader.LCQMCDataloader import LCQMCDataLoader
from transformers.modeling_albert import AlbertConfig

from utils.train import train
from utils.evaluate import evaluate
from torch.optim import lr_scheduler
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
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=1e-8)
    best = 0
    e_t = 0
    while (True):
        epoch_loss = train(net, train_data_loader, optimizer, device)
        print("epoch loss", epoch_loss)
        evalepochloss, acc = evaluate(net, dev_data_loader, device)
        print("eval eopch loss{0}  eval acc {1}".format(evalepochloss,acc))
        scheduler.step()
        if acc > best:
            best = acc
            e_t = 0
            net.save_pretrained(
                "/home/longred/BertForSentenceSimilarity/output/LCQMC/albert")
            print("model save")
        e_t += 1
        if e_t > 15:
            break
