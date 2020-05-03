from tqdm import tqdm



def train(net, train_loader, optim, device,is_pair = False):
    loader = tqdm(train_loader)
    sum_loss = 0
    num = 0
    net.train()
    for i in loader:
        if is_pair:
            loss, out = net(input_ids=i.input_1.to(device),
                            attention_mask=None,
                            token_type_ids=i.input_2.to(device),
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
        loss.backward()
        sum_loss += loss.item()
        num += len(i)
        optim.step()
        optim.zero_grad()
        loader.set_postfix(loss=loss.item())
    return sum_loss/num
