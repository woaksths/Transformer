from __future__ import division
import logging
import os
import random
import time

import torch
import torchtext
from torch import optim
from loss import NLLLoss
from evaluator import Evaluator
from optim import NoamOpt, get_std_opt, Optimizer
from models import Batch, subsequent_mask


class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.
    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss: loss for training, (default: NLLLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
    """
    def __init__(self, expt_dir='experiment', loss=NLLLoss(), batch_size=64,
                 random_seed=None,
                 checkpoint_every=100, print_every=100, input_vocab=None, output_vocab=None):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
            
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.src_pad_idx = input_vocab.stoi['<pad>']
        self.tgt_pad_idx = output_vocab.stoi['<pad>']
        
        
    def _train_batch(self, model, src, tgt, tgt_y, src_mask, tgt_mask):
        # Forward propagation        
        decoder_outputs = model(src, tgt, src_mask, tgt_mask)
        decoder_outputs = model.generator(decoder_outputs)

        # Get loss 
        loss, n_word, n_correct = self.loss.cal_loss(decoder_outputs.contiguous().
                                                     view(-1, decoder_outputs.size(2)),tgt_y.contiguous().view(-1))
        
        # Backward propagation
        self.optimizer.optimizer.zero_grad()
        loss /= n_word
        loss.backward()
        self.optimizer.step()
        
        return loss.item()*n_word, n_word, n_correct 

    
    def _train_epoches(self, data, model, n_epochs, start_epoch, start_step, dev_data=None):
        log = self.logger
        device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=device, repeat=False, shuffle=True)        
        
        steps_per_epoch = len(batch_iterator)
        total_steps = steps_per_epoch * n_epochs
        
        step = start_step
        step_elapsed = 0
        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch
        n_word_total = 0
        n_word_correct =0
        
        for epoch in range(start_epoch, n_epochs + 1):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            batch_generator = batch_iterator.__iter__()
            # consuming seen batches from previous training
            for _ in range((epoch - 1) * steps_per_epoch, step):
                next(batch_generator)

            model.train(True)
            for batch in batch_generator:
                step += 1
                step_elapsed += 1

                input_variables, _ = getattr(batch, 'src')
                target_variables = getattr(batch, 'tgt')

                batch_obj = Batch(input_variables, target_variables, self.tgt_pad_idx)
                loss, n_word, n_correct = self._train_batch(model,  batch_obj.src,  batch_obj.tgt, batch_obj.tgt_y,
                                                            batch_obj.src_mask,  batch_obj.tgt_mask)
                
                # Record average loss
                epoch_loss_total += loss
                n_word_total += n_word
                n_word_correct += n_correct
                
                if step%self.print_every == 0 and step_elapsed > self.print_every:
                    log_msg = 'Progress: %d%%, Train %s: %.4f' % (
                        step / total_steps * 100,
                        self.loss.name,
                        loss/n_word)
                    log.info(log_msg)
                    
                # Checkpoint
                if step % self.checkpoint_every == 0 or step == total_steps:
                    # save model
                    pass

            if step_elapsed == 0: continue

            epoch_loss_avg = epoch_loss_total / n_word_total
            accuracy = n_word_correct / n_word_total
            n_word_total = 0
            epoch_loss_total = 0
            n_word_correct =0
            
            log_msg = "Finished epoch %d: Train %s: %.4f accuracy %.3f" % (epoch, self.loss.name, epoch_loss_avg, accuracy)
            
            if dev_data is not None:
                dev_loss,accuracy= self.evaluator.evaluate(model, dev_data)
                log_msg += ", Dev %s: %.4f, Accuracy: %.4f" % (self.loss.name, dev_loss, accuracy)
                model.train(mode=True)
            else:
                pass
            
            log.info(log_msg)

            
    def train(self, model, data, num_epochs=5,
              resume=False, dev_data=None,
              optimizer=None, teacher_forcing_ratio=0):
        """ Run training for a given model.
        Args:
            model: model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data: dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data: dev Dataset (default None)
            optimizer: optimizer for training (default: Adam+Warmup)
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
        Returns:
            model (transformer): trained model
        """
        # If training is set to resume
        if resume:
            pass
        else:
            start_epoch = 1
            step = 0
            self.optimizer = optimizer

        self.logger.info("Optimizer: %s" % (self.optimizer.optimizer))
        self._train_epoches(data, model, num_epochs,
                            start_epoch, step, dev_data=dev_data)
        return model