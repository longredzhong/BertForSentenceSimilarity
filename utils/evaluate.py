from tqdm import tqdm
import torch
from sklearn.metrics.classification import accuracy_score
from utils.util import seq_len_to_mask

def evaluate(net, evaluate_loader, device,is_pair=False):
    loader = tqdm(evaluate_loader)
    sum_loss = 0
    num = 0
    num_corrects = 0
    net.eval()
    with torch.no_grad():
        for i in loader:
            if is_pair:
                loss, out = net(
                                input_ids=i.input_ids.to(device),
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
            label = i.label
            out = out.squeeze(1)
            out_list = []
            for j in out.tolist():
                if j>0.5:
                    out_list.append(1)
                else:
                    out_list.append(0)
            num_corrects += accuracy_score(
                label.tolist(), out_list, normalize=False)
            sum_loss += loss.item()
            num += len(i)
            loader.set_postfix(loss=loss.item())
    return sum_loss/len(evaluate_loader), num_corrects/num
