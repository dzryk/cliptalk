import importlib
import torch
import pytorch_lightning as pl
import numpy as np

from CLIP import clip
from torch import nn
from transformers import GPT2Tokenizer, AutoModelForCausalLM


def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs


class Model(pl.LightningModule):
    """Model consists of CLIP + GPT."""
    def __init__(self,
                 args=None,
                 indices=None,
                 indices_data=None):
        super(Model, self).__init__()
        GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
        self.save_hyperparameters(args, ignore='index')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=self.hparams.logdir)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.perceiver = clip.load(self.hparams.clip_model, jit=False)[0]
        self.indices = indices
        self.indices_data = indices_data

        self.model = AutoModelForCausalLM.from_pretrained(self.hparams.gpt,
            cache_dir=self.hparams.logdir, pad_token_id=self.tokenizer.eos_token_id)

        # Import the task
        task = importlib.import_module(f'tasks.{self.hparams.task}')
        self.build_table = task.build_table

        # Shut off gradients
        for pname, p in self.model.named_parameters():
            pname = pname.lower()
            if self.hparams.ft in pname:
                p.requires_grad = True
            else:
                p.requires_grad = False
        for pname, p in self.perceiver.named_parameters():
            p.requires_grad = False
    
    def compute_loss(self, x, y, ctx=None):
        x = self.build_table(x,
                             y=y,
                             ctx=ctx,
                             tokenize=clip.tokenize,
                             perceiver=self.perceiver,
                             indices=self.indices,
                             indices_data=self.indices_data,
                             device=self.device,
                             knn=self.hparams.knn)
        inputs = self.tokenizer(x,
                                padding='max_length',
                                truncation=True,
                                max_length=self.hparams.maxlen,
                                return_tensors='pt')

        # Labels. This is a bit hacky but works for GPT2 tokenizer.
        labels = inputs.input_ids.numpy().copy()
        zipped = [zip(masks, labels) for masks, labels in zip(inputs.attention_mask, labels)]
        labels = np.array(
            [[-100 if mask == 0 else token for mask, token in mask_tokens] for mask_tokens in zipped])
        for idx, entry in enumerate(labels):
            end = np.where(entry == 91)[0][0]
            labels[idx][:end+1] = -100
        
        loss = self.model(input_ids=torch.tensor(
                              inputs.input_ids, device=self.device),
                          attention_mask=torch.tensor(
                              inputs.attention_mask, device=self.device),
                          labels=torch.tensor(labels, device=self.device)).loss
        return loss

    def unpack(self, batch):
        if self.hparams.task == 'im2txt':
            x, y = batch
            return x, y, None
        elif self.hparams.task == 'txt2txt':
            x, _ = batch
            if '\t' in x[0]:
                x = [l.split('\t') for l in x]
                txt = [t[0] for t in x]
                ctx = [t[1] for t in x]
                return txt, txt, ctx
            else:
                return x, x, None
            
    def training_step(self, batch, batch_idx):
        x, y, ctx = self.unpack(batch)
        loss = self.compute_loss(x, y, ctx)
        self.log('loss', loss, on_step=True, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y, ctx = self.unpack(batch)
        loss = self.compute_loss(x, y, ctx)
        self.log('vloss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lrate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.hparams.tmax)
        return [optimizer], [scheduler]