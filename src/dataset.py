import random
from typing import Optional, Tuple, List, Iterator, Dict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

from src.utils import read_from_input_file, flatten, chunks


class BartDataset(IterableDataset):

    def __init__(self, input_path: str, bart_model: str, max_tokens_per_batch: int,
                 chunk_size: int = 4000, gold_path: str = None,
                 input_dataset_path: str = None) -> None:

        super(BartDataset, self).__init__()

        self.input_path = input_path
        self.tokenizer = AutoTokenizer.from_pretrained(bart_model, is_fast=True)
        self.max_tokens_per_batch = max_tokens_per_batch
        self.dataset_store = []
        self.chunk_size = chunk_size
        self.target_token_start = '<t>'
        self.target_token_end = '</t>'

        self.log_input = None 
        if input_dataset_path:
            self.log_input = input_dataset_path 
            print(f'saving input corpus in {self.log_input}')

        self.output_vocabulary = set()
        if gold_path:
            for line in open(gold_path):
                words = line.strip().split(' :: ')[-1]
                for word in words.split(';'):
                    w = ' '.join(word.split()[:-1])
                    if w != ' ':
                        self.output_vocabulary.add(w)


    def encode(self, source: str, target: Optional[str] = None) -> Tuple[List[int], Optional[List[int]]]:

        if target:

            sample = self.tokenizer.prepare_seq2seq_batch([source], tgt_texts=[target], return_tensors='pt')
            return sample['input_ids'][0], sample['labels'][0]

        sample = self.tokenizer.prepare_seq2seq_batch([source], return_tensors='pt')
        return sample['input_ids'][0]

    def __init_dataset(self) -> None:

        if self.log_input:
            out_f = open(self.log_input, 'w')

        is_training = False

        for instance in read_from_input_file(self.input_path):
            sentence_words = instance.sentence.split(" ")

            pre = " ".join(sentence_words[:instance.target_idx[0]]) + " " + self.target_token_start
            mid = " ".join(sentence_words[instance.target_idx[0]: instance.target_idx[-1] + 1])
            post = self.target_token_end + " " + " ".join(sentence_words[instance.target_idx[-1] + 1:])

            str_input = f'{pre}{mid}{post}'

            if instance.gold:
                is_training = True
                instance.gold = {word: score for word, score in instance.gold.items() if word != ''}

                if len(instance.gold) == 0:
                    continue

                sorted_gold = sorted([(k, v) for k, v in instance.gold.items()], key=lambda x: x[1], reverse=True)

                str_label = ', '.join([label.replace('_', ' ') for label, score in sorted_gold])
                
                if self.log_input:
                    out_f.write(f'{instance.target}\t{str_input}\t{str_label}\n')
                
                encoded_input, encoded_labels = self.encode(str_input, str_label)
                self.dataset_store.append((encoded_input, encoded_labels, instance))

            else:
                encoded_input = self.encode(str_input)
                self.dataset_store.append((encoded_input, instance))

        if is_training:
            self.dataset_store = sorted(self.dataset_store, key=lambda x: len(x[0]) + len(x[1]) + random.randint(0, 10))

        else:
            self.dataset_store = sorted(self.dataset_store, key=lambda x: len(x[0]) + random.randint(0, 10))

        self.dataset_store = list(chunks(self.dataset_store, self.chunk_size))
        random.shuffle(self.dataset_store)
        self.dataset_store = flatten(self.dataset_store)

        if self.log_input:
            out_f.close()

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:

        if len(self.dataset_store) == 0:
            self.__init_dataset()

        current_batch = []

        def output_batch() -> Dict[str, torch.Tensor]:

            input_ids = pad_sequence([x[0] for x in current_batch], batch_first=True,
                                     padding_value=self.tokenizer.pad_token_id)
            input_padding_mask = input_ids != self.tokenizer.pad_token_id

            if len(current_batch[0]) == 3:
                labels = pad_sequence([x[1] for x in current_batch], batch_first=True,
                                      padding_value=self.tokenizer.pad_token_id)

                labels_padding_mask = labels != self.tokenizer.pad_token_id

            else:
                labels = None
                labels_padding_mask = None

            instances = [x[-1] for x in current_batch]

            batch = {
                "source": input_ids,
                "source_padding_mask": input_padding_mask,
                "metadata": instances,
                "target": labels,
                "target_padding_mask": labels_padding_mask,
            }

            return batch

        for element in self.dataset_store:

            if len(element) == 3:
                encoded_source, encoded_target, instance = element

                future_source_tokens = max(
                    max([encoded_source.size(0) for encoded_source, *_ in current_batch], default=0),
                    len(encoded_source)
                ) * (len(current_batch) + 1)

                future_target_tokens = max(
                    max([encoded_target.size(0) for _, encoded_target, _ in current_batch], default=0),
                    len(encoded_target)
                ) * (len(current_batch) + 1)

                if future_source_tokens + future_target_tokens > self.max_tokens_per_batch:

                    if len(current_batch) == 0:
                        continue

                    yield output_batch()
                    current_batch = []

                current_batch.append((encoded_source, encoded_target, instance))

            else:
                encoded_source, instance = element

                future_source_tokens = max(
                    max([encoded_source.size(0) for encoded_source, *_ in current_batch], default=0),
                    len(encoded_source)
                ) * (len(current_batch) + 1)

                if future_source_tokens > self.max_tokens_per_batch:

                    if len(current_batch) == 0:
                        continue

                    yield output_batch()
                    current_batch = []

                current_batch.append((encoded_source, instance))

        if len(current_batch) != 0:
            yield output_batch()
