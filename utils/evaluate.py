from tqdm import tqdm
import torch
from sklearn.metrics.classification import accuracy_score

def evaluate(net, evaluate_loader, device):
    loader = tqdm(evaluate_loader)
    sum_loss = 0
    num = 0
    num_corrects = 0
    net.eval()
    with torch.no_grad():
        for i in loader:
            loss, out = net(input_ids=i.input_1.to(device),
                            attention_mask=None,
                            token_type_ids=i.input_2.to(device),
                            position_ids=None,
                            head_mask=None,
                            labels=i.label.to(device).float()
                            )
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
    return sum_loss/num, num_corrects/num
