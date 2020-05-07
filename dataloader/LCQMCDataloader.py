from torchtext.data import BucketIterator, Dataset, Example, Field, Iterator
from torchtext.data.field import RawField
from utils.util import sequence_padding,toTensor
from utils.tokenizers import Tokenizer

class LCQMCDataset(Dataset):
    def __init__(self, path, fields, tokenizer,  length=None):
        examples = []
        with open(path, 'r') as f:
            data = f.readlines()
        for i in range(len(data)):
            data[i] = data[i].strip().split('\t')
        print('read data from{}'.format(path))
        for i in data:
            guid = i
            input_1 = tokenizer.encode(i[0],max_length=length)[0]
            input_2 = tokenizer.encode(i[1], max_length=length)[0]
            if length is not None:
                padding_length = length-len(input_1)
                input_1 += [0] * padding_length
                padding_length = length-len(input_2)
                input_2 += [0] * padding_length
            label = int(i[2])
            examples.append(Example.fromlist([guid,input_1,len(input_1), input_2,len(input_2), label], fields))
        super().__init__(examples, fields)

class LCQMCDataset_pair(Dataset):
    def __init__(self, path, fields, tokenizer,length=None):
        examples = []
        with open(path, 'r') as f:
            data = f.readlines()
        for i in range(len(data)):
            data[i] = data[i].strip().split('\t')
        print('read data from{}'.format(path))
        for i in data:
            guid = i
            input_ids, segment_ids = tokenizer.encode(
                i[0], i[1], max_length=length)
            attention_mask = [1]*len(input_ids)
            label = int(i[2])
            if length is not None:
                padding_length=length-len(input_ids)
                input_ids += [0] * padding_length
                segment_ids += [0]*padding_length
                attention_mask += [0]*padding_length
            examples.append(Example.fromlist(
                [guid,input_ids, segment_ids, attention_mask, len(input_ids), label], fields))
        super().__init__(examples, fields)


def LCQMCDataLoader(data_path, vocab_path,batch_size = 32,is_pair=False,length = None):

    tokenizer = Tokenizer(vocab_path)
    if is_pair:
        fields = [
            ("guid", RawField()),
            ("input_ids", RawField(postprocessing=sequence_padding)),
            ("input_token_type", RawField(postprocessing=sequence_padding)),
            ("input_attention_mask", RawField(postprocessing=sequence_padding)),
            ("input_len", RawField(postprocessing=toTensor)),
            ("label", RawField(postprocessing=toTensor))
        ]
        dataset = LCQMCDataset_pair(
            path=data_path, fields=fields, tokenizer=tokenizer, length=length)
    else:
        fields = [
            ("guid", RawField()),
            ("input_1", RawField(postprocessing=sequence_padding)),
            ("input_1_len", RawField(postprocessing=toTensor)),
            ("input_2", RawField(postprocessing=sequence_padding)),
            ("input_2_len", RawField(postprocessing=toTensor)),
            ("label", RawField(postprocessing=toTensor))
        ]
        dataset = LCQMCDataset(
            path=data_path, fields=fields, tokenizer=tokenizer, length=length)

    dataloader = Iterator(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
