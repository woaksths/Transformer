from __future__ import print_function, division

import torch
import torchtext

from loss import NLLLoss
from models import Batch, subsequent_mask

class Evaluator(object):
    """ Class to evaluate models with given datasets.
    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.
        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against
        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields['tgt'].vocab
        pad = tgt_vocab.stoi[data.fields['tgt'].pad_token]

        with torch.no_grad():
            for batch in batch_iterator:
                input_variables, _  = getattr(batch, 'src')
                target_variables = getattr(batch, 'tgt')

                batch_obj = Batch(input_variables, target_variables, 1)
                
                decoder_outputs = model(batch_obj.src, batch_obj.tgt,
                                        batch_obj.src_mask, batch_obj.tgt_mask)
                
                decoder_outputs = model.generator(decoder_outputs)

                # Evaluation
                for step in range(decoder_outputs.size(1)):
                    target = batch_obj.tgt_y[:, step]
                    loss.eval_batch(decoder_outputs[:,step,:], target)
        return loss.get_loss()