#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from dataset import Wiki2Dataset, ToTensor, seq_collate_fn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm
import os
import sys

import pandas as pd
import time
import math

from model import LM
from utils import create_argument_parser, get_arg_groups, registry_values, DotMap

class Trainer(object):

    def __init__(self, model, train_corpus_size, config):
        self.model = model
        self.config = config
        print(f"config: {config}")
        self.device = torch.device(self.config.device)
        self.criterion = nn.CrossEntropyLoss()
        # for now we'll assume SGD
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=self.config.momentum)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.gamma)
        self.train_corpus_size = train_corpus_size
        self.writer = SummaryWriter('runs/lm')

    def forward_pass(self, batch, hidden):
        source = Variable(batch['sources'].to(self.device), requires_grad=False)
        targets = Variable(batch['targets'].to(self.device), requires_grad=False)
        output, hidden = self.model(source, hidden)
        return self.criterion(output, targets.reshape(-1)), hidden


    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_value)

    def train(self, dataloader, epoch):
        start_time = time.time()
        total_loss = 0
        self.model.train()
    
        for idx, batch in enumerate(tqdm(dataloader)):
            bsz = batch['sources'].size(0)
            hx, ctx = self.model.init_hidden(bsz)
            hidden = hx.to(self.device), ctx.to(self.device)
            loss, hidden = self.forward_pass(batch, hidden)
            self.step(loss)
        
            total_loss += loss.item()

            if idx % self.config.log_interval == 0 and idx > 0:
                cur_loss = total_loss / self.config.log_interval
                elapsed = time.time() - start_time
                self.writer.add_scalar('training_loss', cur_loss, epoch * len(dataloader) + idx)

                #print(f'| epoch {epoch:3d} | {idx:5d}/{len(dataloader):3d} batches |' 
                #  f'lr {self.config.lr:4.8f} | ms/batch {elapsed * 1000 / self.config.log_interval:5.2f} | '
                #    f'loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}')
                total_loss = 0
                start_time = time.time()

    def evaluate(self, dataloader):
        self.model.eval()
        val_loss = 0.
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader)):
                bsz = batch['sources'].size(0)
                hx, ctx = self.model.init_hidden(bsz)
                hidden = hx.to(self.device), ctx.to(self.device)
                loss, _ = self.forward_pass(batch, hidden)
                val_loss += loss.item()
        return val_loss / len(dataloader)


    def run(self, train_dataloader, val_dataloader):

        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)
            os.mkdir(os.path.join(self.config.checkpoint_dir, 'best_models'))
        
    
        print(f"| Trainable Parameters: {self.model.num_params:6,d}")
        self.model = self.model.to(self.device)
        #model = nn.DataParallel(model)
        print(f"| learning_rate: {self.config.lr: 2.4f} | gamma: {self.config.gamma: 2.4f} | momentum: {self.config.momentum: 2.4f} | epochs: {self.config.num_epochs: 3d}")
    
        best_val_loss = None
        ppls = []
        dataset = iter(train_dataloader)
        batch = dataset.next()
        source = Variable(batch['sources'].to(self.device), requires_grad=False)
        bsz = source.size(0)
        hx, ctx = self.model.init_hidden(bsz)
        hidden = hx.to(self.device), ctx.to(self.device)
        self.writer.add_graph(self.model, (source, hidden))
        self.writer.add_text("model", f"Embedding size: {self.config.hidden_size}, Number of Layers: {self.config.num_layers}, Trainable Parameters: {self.model.num_params:6,d}, model: {str(self.model)}")
        self.writer.add_text("data", f"Vocab Size: {self.config.input_size}, Train Corpus Size: {self.config.train_corpus_size}, Number of sentences in Train Dataset: {len(train_dataloader)*bsz}")
        self.writer.add_text("training", f"Number of epochs: {self.config.num_epochs: 3d}")
    
        for epoch in tqdm(range(self.config.num_epochs)):
            epoch_start_time = time.time()
            lr = self.scheduler.get_last_lr()[0]
            self.train(train_dataloader, epoch)
        
            val_loss = self.evaluate(val_dataloader)
            valid_ppl = math.exp(val_loss)
            self.writer.add_scalar('validation_loss', val_loss, epoch * len(val_dataloader))
            self.writer.add_scalar('validation_perplexity', valid_ppl, epoch * len(val_dataloader))
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, valid_ppl))
            print('-' * 89)
            if best_val_loss:
                print(f"| best_val_loss: {best_val_loss: 2.4f} | val_loss: {val_loss: 2.4f}")
        
            ppls.append(valid_ppl)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                state = {
                    'lr': lr,
                    'ppl': valid_ppl,
                    'loss': val_loss,
                    'epoch': epoch,
                    'model': self.model
                }
                state['config'] = self.config.to_dict()
                with open(os.path.join(f'models/epoch{epoch}_lr{lr}_ppl{valid_ppl}_loss{val_loss}.ckpt'), 'wb') as f:
                    torch.save(state, f)
                with open('models/best_models/model.ckpt','wb') as f:
                    torch.save(state, f)
                best_val_loss = val_loss
                self.writer.add_scalar('best_validation_loss', best_val_loss, epoch * len(val_dataloader))
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                self.scheduler.step()
         
        return ppls
        
parser = create_argument_parser(registry_values)
args = parser.parse_args()
arg_groups = get_arg_groups(parser, args)
print(arg_groups)

wiki_dataset = Wiki2Dataset(args.data_dir)
train_corpus_size = wiki_dataset.train_dataset.corpus_size
vocab_size = len(wiki_dataset.vocab)
model = LM(vocab_size, args.hidden_size, num_layers=args.num_layers, 
           dropout=args.dropout_p)

print(f"| Train Num Tokens: {wiki_dataset.train_dataset.corpus_size: 4d} | Train Num Examples: {len(wiki_dataset.train_dataset): 4d}")
print(f"| Valid Num Tokens: {wiki_dataset.valid_dataset.corpus_size: 4d} | Valid Num Examples: {len(wiki_dataset.valid_dataset): 4d}")
print(f"| Test Num Tokens: {wiki_dataset.test_dataset.corpus_size: 4d} | Test Num Examples: {len(wiki_dataset.test_dataset): 4d}")

train_dataloader = DataLoader(wiki_dataset.train_dataset, shuffle=args.shuffle_data, 
                              batch_size=args.train_batch_size, 
                              num_workers=args.num_workers, 
                              collate_fn=seq_collate_fn, 
                              drop_last=args.drop_last)

valid_dataloader = DataLoader(wiki_dataset.valid_dataset, shuffle=args.shuffle_data, 
                              batch_size=args.valid_batch_size, 
                              num_workers=args.num_workers, 
                              collate_fn=seq_collate_fn, 
                              drop_last=args.drop_last)


input_size = len(wiki_dataset.vocab)
args.input_size = input_size
config = DotMap(vars(args))
trainer = Trainer(model, train_corpus_size, config)
ppls = trainer.run(train_dataloader, valid_dataloader)








