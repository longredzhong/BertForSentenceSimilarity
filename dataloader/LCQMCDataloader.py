from torchtext.data import BucketIterator, Dataset, Example, Field, Iterator
from torchtext.data.field import RawField
from utils.util import sequence_padding,toTensor
from utils.tokenizers import Tokenizer

class LCQMCDataset(Dataset):
    def __init__(self,path,fields,tokenizer,):
        examples = []
        with open(path, 'r') as f:
            data = f.readlines()
        for i in range(len(data)):
            data[i] = data[i].strip().split('\t')
        print('read data from{}'.format(path))
        for i in data:
            input_1 = tokenizer.encode(i[0],max_length=512)[0]
            input_2 = tokenizer.encode(i[1], max_length=512)[0]
            label = int(i[2])
            examples.append(Example.fromlist([input_1, input_2, label], fields))
        super().__init__(examples, fields)

class LCQMCDataset_pair(Dataset):
    def __init__(self, path, fields, tokenizer,):
        examples = []
        with open(path, 'r') as f:
            data = f.readlines()
        for i in range(len(data)):
            data[i] = data[i].strip().split('\t')
        print('read data from{}'.format(path))
        for i in data:
            input_ids, segment_ids = tokenizer.encode(
                i[0], i[1], max_length=512)
            label = int(i[2])
            examples.append(Example.fromlist(
                [input_ids, segment_ids, label], fields))
        super().__init__(examples, fields)


def LCQMCDataLoader(data_path, vocab_path,batch_size = 32,is_pair=False):
    fields = [
        ("input_1", RawField(postprocessing=sequence_padding)),
        ("input_2", RawField(postprocessing=sequence_padding)),
        ("label", RawField(postprocessing=toTensor))
    ]
    tokenizer = Tokenizer(vocab_path)
    if is_pair:
        dataset = LCQMCDataset_pair(path=data_path,fields=fields,tokenizer=tokenizer)
    else:
        dataset = LCQMCDataset(
            path=data_path, fields=fields, tokenizer=tokenizer)

    dataloader = Iterator(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
