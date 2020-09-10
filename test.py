from utils import Checkpoint
from torch.autograd import Variable
from models import subsequent_mask
import torch
import argparse
from FAdo.reex import *

def get_test_data(fname):
    src_set = []
    tgt_set = []
    with open(fname, 'r') as rf:
        dataset = rf.read().split('\n')
        for d in dataset:
            if d.strip() =='':
                continue
            src, tgt = d.split('\t')[0], d.split('\t')[1]
            src_set.append(src)
            tgt_set.append(tgt)

    return src_set, tgt_set


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.ones(1,1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len-1):
            out = model.decode(memory.cuda(), src_mask.cuda(), Variable(ys).cuda(),
                               Variable(subsequent_mask(ys.size(1)).type_as(src.data)).cuda())
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim = 1)
            next_word = next_word.data[0]
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


parser = argparse.ArgumentParser(description="Type trained model")
parser.add_argument('--checkpoint', help='Path to trained model')
parser.add_argument('--test_path', help='Path to test data')
opt = parser.parse_args()

checkpoint = Checkpoint.get_latest_checkpoint(opt.checkpoint)
transformer = Checkpoint.load(checkpoint)
input_vocab = transformer.input_vocab
output_vocab = transformer.output_vocab
test_src, test_tgt = get_test_data(opt.test_path)


dfa_equal = 0 
string_equal = 0 
invalid_regex = 0

for idx, (t_src, t_tgt)  in enumerate(zip(test_src, test_tgt)):
    source = t_src.replace(' ','')
    target = t_tgt.replace(' ','')
    src = torch.LongTensor([input_vocab.stoi[char] for char in t_src.split(' ')]).unsqueeze(0)
    src_mask = Variable(torch.ones(1, 1, src.size(1)))
    out = greedy_decode(transformer.model, src.cuda(), src_mask.cuda(), max_len=100, start_symbol=output_vocab.stoi['<sos>'])
    out = out.view(-1)
    predict =''
    
    for i in range(1, out.size(0)):
        if output_vocab.itos[out[i]] == '<eos>':
            break
        predict +=  output_vocab.itos[out[i]]
       
    print('source:', source)
    print('target:', target)
    print('predict:', predict)
    print()
    
    try:
        # DFA equivalence 
        tgt_dfa = str2regexp(target).toDFA()
        pred_dfa = str2regexp(predict).toDFA()
        if tgt_dfa == pred_dfa:
            dfa_equal += 1

        # String Equal
        if target == predict:
            string_equal +=1 
    except:
        invalid_regex +=1 
        
print('total test data:{}, dfa equal:{}, string equal:{}'.format(len(test_src), dfa_equal, string_equal))
print('invalid regex {}'.format(invalid_regex))