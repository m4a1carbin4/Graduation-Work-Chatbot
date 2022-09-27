import os
import shutil
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from NER_utils import compute_metrics, get_labels, get_test_texts, show_report, MODEL_CLASSES

logger = logging.getLogger(__name__)

class Trainer(object):
    
    def __init__(self,train_dataloader=None,test_datalodaer=None,train_label=None,test_label=None,**kwargs):

        self.model_dir = '/NER'
        self.train_dl = train_dataloader
        self.test_dl = test_datalodaer

        self.seed = kwargs.get('seed',42)
        self.train_batch_size = kwargs.get('train_batch_size',32)
        self.eval_batch_size = kwargs.get('eval_batch_size',64)
        self.max_seq_len = kwargs.get('max_seq_len',50)
        self.learning_rate = kwargs.get('learning_rate',5e-5)
        self.num_train_epochs = kwargs.get("num_train_epochs", 20.0)
        self.weight_decay = kwargs.get("weight_decay", 0.0)
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps',1)
        self.adam_epsilon = kwargs.get("adam_epsilon", 1e-8)
        self.max_grad_norm = kwargs.get("max_grad_norm",1.0)
        self.max_steps = kwargs.get("max_steps", -1)
        self.warmup_steps = kwargs.get("warmup_steps", 0)

        self.logging_steps = kwargs.get('logging_steps',1000)
        self.save_steps = kwargs.get('save_steps', 1000)

        self.train_label_lst = train_label
        self.test_label_lst = test_label

        self.train_num = len(self.train_label_lst)
        self.test_num = len(self.test_label_lst)

        self.pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

        self.config_class, self.model_class, _ = MODEL_CLASSES['kobert']

        self.config = self.config_class.from_pretrained('monologg/kobert')
        self.model = self.model_class.from_pretrained('monologg/kobert', config=self.config)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train(self):

        if self.max_steps > 0:
            t_total = self.max_steps
            self.num_train_epochs = self.max_steps // (len(self.train_dl) // self.gradient_accumulation_steps) + 1
        else:
            t_total = len(self.train_dl) // self.gradient_accumulation_steps * self.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dl))
        logger.info("  Num Epochs = %d", self.num_train_epochs)
        logger.info("  Total train batch size = %d", self.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.logging_steps)
        logger.info("  Save steps = %d", self.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(self.train_dl, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                print(batch[0])
                print(batch[1])
                print(batch[2])
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[0],
                          'labels': batch[1]}
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.logging_steps == 0:
                        self.evaluate("test", global_step)

                    if self.args.save_steps > 0 and global_step % self.save_steps == 0:
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step
    
    def evaluate(self, mode, step):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if self.args.model_type != 'distilkobert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Slot prediction
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Slot result
        preds = np.argmax(preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.label_lst)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(slot_label_map[out_label_ids[i][j]])
                    preds_list[i].append(slot_label_map[preds[i][j]])

        if self.args.write_pred:
            if not os.path.exists(self.pred_dir):
                os.mkdir(self.pred_dir)

            with open(os.path.join(self.pred_dir, "pred_{}.txt".format(step)), "w", encoding="utf-8") as f:
                for text, true_label, pred_label in zip(self.test_texts, out_label_list, preds_list):
                    for t, tl, pl in zip(text, true_label, pred_label):
                        f.write("{} {} {}\n".format(t, tl, pl))
                    f.write("\n")

        result = compute_metrics(out_label_list, preds_list)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info("\n" + show_report(out_label_list, preds_list))  # Get the report for each tag result

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.model_dir)

        # Save training arguments together with the trained model
        torch.save(os.path.join(self.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.model_dir)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")

