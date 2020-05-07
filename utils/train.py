from tqdm import tqdm
import torch
from .util import seq_len_to_mask
def train(net, train_loader, optim, scheduler, device, is_pair= False):
    loader = tqdm(train_loader)
    sum_loss = 0
    num = 0
    net.train()
    for i in loader:
        if is_pair:
            loss, out = net(input_ids=i.input_ids.to(device),
                            attention_mask=i.input_attention_mask.to(device),
                            token_type_ids=i.input_token_type.to(device),
                            position_ids=None,
                            head_mask=None,
                            labels=i.label.to(device).float()
                            )
        else:
            loss, out = net(input_1=i.input_1.to(device),
                            input_1_lengths=i.input_1_len.to(device),
                            input_2=i.input_2.to(device),
                            input_2_lengths=i.input_2_len.to(device),
                            labels=i.label.to(device).float()
                            )
        loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        sum_loss += loss.item()
        num += len(i)
        optim.step()
        scheduler.step()
        optim.zero_grad()
        loader.set_postfix(loss=loss.item())
    return sum_loss/len(train_loader)
