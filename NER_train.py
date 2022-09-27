import os
import shutil
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer

from NER_utils import compute_metrics, get_labels, get_test_texts, show_report, MODEL_CLASSES

logger = logging.getLogger(__name__)

class Trainer(object):
    
    def __init__(self,dataset,label,**kwargs):

        self.model_dir = '/NER'

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
        

        self.train_data_sentence = [sent.split() for sent in dataset['str'].values]
        self.train_data_label = [tag.split() for tag in dataset['label'].values]

        self.labels = [k for k in label.keys()]

        self.tag_to_index = {tag: index for index, tag in enumerate(self.labels)}
        self.index_to_tag = {index: tag for index, tag in enumerate(self.labels)}

        self.train_num = len(self.train_data_sentence)

        self.pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

        self.config_class, self.model_class, _ = MODEL_CLASSES['kobert']

        self.config = self.config_class.from_pretrained('monologg/kobert')
        self.model = self.model_class.from_pretrained('monologg/kobert', config=self.config)
        self.tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

        self.train_dataset = self.convert_examples_to_features(self.train_data_sentence, self.train_data_label, max_seq_len=128, tokenizer=self.tokenizer)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def convert_examples_to_features(self,examples, labels, max_seq_len, tokenizer,
                                 pad_token_id_for_segment=0, pad_token_id_for_label=-100):
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        pad_token_id = tokenizer.pad_token_id

        input_ids, attention_masks, token_type_ids, data_labels = [], [], [], []

        for example, label in tqdm(zip(examples, labels), total=len(examples)):
            tokens = []
            labels_ids = []
            for one_word, label_token in zip(example, label):
                subword_tokens = tokenizer.tokenize(one_word)
                tokens.extend(subword_tokens)
                labels_ids.extend([self.tag_to_index[label_token]]+ [pad_token_id_for_label] * (len(subword_tokens) - 1))

            special_tokens_count = 2
            if len(tokens) > max_seq_len - special_tokens_count:
                tokens = tokens[:(max_seq_len - special_tokens_count)]
                labels_ids = labels_ids[:(max_seq_len - special_tokens_count)]

            tokens += [sep_token]
            labels_ids += [pad_token_id_for_label]
            tokens = [cls_token] + tokens
            labels_ids = [pad_token_id_for_label] + labels_ids

            input_id = tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_id)
            padding_count = max_seq_len - len(input_id)

            input_id = input_id + ([pad_token_id] * padding_count)
            attention_mask = attention_mask + ([0] * padding_count)
            token_type_id = [pad_token_id_for_segment] * max_seq_len
            label = labels_ids + ([pad_token_id_for_label] * padding_count)

            assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
            assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_id), max_seq_len)
            assert len(label) == max_seq_len, "Error with labels length {} vs {}".format(len(label), max_seq_len)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)
            data_labels.append(label)

        input_ids = np.array(input_ids, dtype=int)
        attention_masks = np.array(attention_masks, dtype=int)
        token_type_ids = np.array(token_type_ids, dtype=int)
        data_labels = np.asarray(data_labels, dtype=np.int32)

        return (input_ids, attention_masks, token_type_ids, data_labels)

    def train(self):

        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.train_batch_size)

        if self.max_steps > 0:
            t_total = self.max_steps
            self.num_train_epochs = self.max_steps // (len(train_dataloader) // self.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.gradient_accumulation_steps * self.num_train_epochs

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
        logger.info("  Num examples = %d", len(train_dataloader))
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
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
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

