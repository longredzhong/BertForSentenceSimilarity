# %%
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import AdamW
import torch
from model.AlbertForSequenceClassification import AlbertForSequenceClassification
from model.ESIM import ESIM
from dataloader.LCQMCDataloader import LCQMCDataLoader
from transformers.modeling_albert import AlbertConfig

from utils.train import train
from utils.evaluate import evaluate
from torch.optim import lr_scheduler
from torch import nn
if __name__ == "__main__":
    train_data_path = "/home/longred/BertForSentenceSimilarity/dataset/LCQMC/train.txt"
    dev_data_path = "/home/longred/BertForSentenceSimilarity/dataset/LCQMC/dev.txt"
    test_data_path = "/home/longred/BertForSentenceSimilarity/dataset/LCQMC/test.txt"
    vocab_path = "/home/longred/BertForSentenceSimilarity/prev_trained_model/albert_tiny_zh/vocab.txt"

    train_data_loader = LCQMCDataLoader(
        train_data_path, vocab_path, batch_size=1024, is_pair=True, length=80)
    dev_data_loader = LCQMCDataLoader(
        dev_data_path, vocab_path, batch_size=1024, is_pair=True, length=80)
    test_data_loader = LCQMCDataLoader(
        test_data_path, vocab_path, batch_size=1024, is_pair=True, length=80)
    config = AlbertConfig.from_pretrained(
        "/home/longred/BertForSentenceSimilarity/prev_trained_model/albert_tiny_zh/config.json")

    config.num_labels = 1
    # config.hidden_size = 128
    config.dropout = 0.5
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    net = AlbertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path="/home/longred/BertForSentenceSimilarity/prev_trained_model/albert_tiny_zh/pytorch_model.bin",
        config=config).to(device)
    # %%
    learning_rate = 5e-4
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(net.bert.named_parameters())
    linear_param_optimizer = list(net.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': learning_rate},
        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': 0.001},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': 0.001}
    ]
    optimizer = AdamW(params=optimizer_grouped_parameters,
                      lr=learning_rate, correct_bias=False)
    epoch = 60
    total_training_steps = len(train_data_loader)*epoch
    warmup_proportion = 0.05
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_training_steps*warmup_proportion,
        num_training_steps=total_training_steps,
        num_cycles=2
    )
    best = 0
    e_t = 0
    net = nn.DataParallel(net, device_ids=[0, 1])
    for i in range(epoch):
        epoch_loss = train(net, train_data_loader, optimizer,
                           scheduler, device, is_pair=True)
        print(i+1, " epoch loss : ", epoch_loss)
        devepochloss, devacc = evaluate(
            net, dev_data_loader, device, is_pair=True)
        print("eval dev eopch loss {0}  eval acc {1}".format(
            devepochloss, devacc))
        testepochloss, testacc = evaluate(
            net, test_data_loader, device, is_pair=True)
        print("eval test eopch loss {0}  eval acc {1}".format(
            testepochloss, testacc))
        acc = devacc
        if acc > best:
            best = acc
            e_t = 0
            config.acc = best
            model_to_save = (
                net.module if hasattr(net, "module") else net
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(
                "/home/longred/BertForSentenceSimilarity/output/LCQMC/albert/")
            print("model save  acc=", best)
        e_t += 1
        if e_t > 20:
            break
