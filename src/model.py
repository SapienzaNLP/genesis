import collections
import os
from typing import Dict, List, Tuple, Optional

import pytorch_lightning as pl
import torch
import transformers
import wandb
import yaml

from src.metrics import PrecisionAtOne
from src.optimizers import RAdam


class BartModel(pl.LightningModule):

    def __init__(self, config_path: str, *args, **kwargs):
        super().__init__(*args)
        self.save_hyperparameters()

        self.config = yaml.load(open(config_path), Loader=yaml.FullLoader)

        if 'output_name' in self.config['datasets']:
            self.output_name = self.config['datasets']['output_name']

        else:
            self.output_name = ''

        # model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config["model"]["name"])
        self.model = transformers.BartForConditionalGeneration.from_pretrained(self.config["model"]["name"])

        if 'dropout' in kwargs:
            self.model.config.dropout = kwargs['dropout']
            print(f'Using dropout {self.model.config.dropout}')

        if 'encoder_layerdropout' in kwargs:
            self.model.config.encoder_layerdrop = kwargs['encoder_layerdropout']
            print(f'Using encoder layerdropout: {self.model.config.encoder_layerdrop}')

        if 'decoder_layerdropout' in kwargs:
            self.model.config.decoder_layerdrop = kwargs['decoder_layerdropout']
            print(f'Using decoder layerdropout: {self.model.config.decoder_layerdrop}')

        if 'attention_dropout' in kwargs:
            self.model.config.attention_dropout = kwargs['attention_dropout']
            print(f'Using attention dropout: {self.model.config.attention_dropout}')

        if 'activation_dropout' in kwargs:
            self.model.config.activation_dropout = kwargs['activation_dropout']
            print(f'Using activation dropout: {self.model.config.activation_dropout}')

        # optimiser
        self.optimiser_dict = self.config["optimiser"]

        # metrics
        self.accuracy_train = pl.metrics.Accuracy()
        self.accuracy_dev = pl.metrics.Accuracy()
        self.val_prec = PrecisionAtOne()

        # generation
        self.generation_parameters = self.config["generation_parameters"]

        self.split_seq_token = '\n'
        self.generated_batches = []

        self.output_vocabulary = set()

        gold_path = os.path.join(self.config["paths"]["scorer_dir"],
                                 f'{self.config["datasets"]["test"]}_gold.txt')

        for line in open(gold_path):
            words = line.strip().split(' :: ')[-1]
            for word in words.split(';'):
                w = ' '.join(word.split()[:-1])
                if w != ' ' and w!='':
                    self.output_vocabulary.add(w)


    def forward(self, source: torch.Tensor, source_padding_mask: torch.Tensor, metadata: List,
                target: Optional[torch.Tensor]=None,
                target_padding_mask: Optional[torch.Tensor]=None):

        if any(t is not None for t in target):
            labels = target[:, 1:].contiguous()
            labels_padding_mask = target_padding_mask[:, 1:].contiguous()
            labels[~labels_padding_mask] = -100

            output = self.model(
                input_ids=source,
                attention_mask=source_padding_mask,
                labels=labels,
                return_dict=True
            )

            return {
                'logits': output['logits'],
                'loss': output['loss']
            }

        else:
            output = self.model(
                input_ids=source,
                attention_mask=source_padding_mask,
                return_dict=True
            )

            return {
                'logits': output['logits']
            }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:

        output = self.forward(**batch)
        self.log('train_loss', output['loss'], prog_bar=False, on_epoch=True, on_step=False)
        return output['loss']

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):

        output = self.forward(**batch)

        logits = output["logits"]
        labels = batch["target"][:, 1:].contiguous()

        # loss
        self.log('val_loss', output['loss'], prog_bar=False, on_step=False, on_epoch=True)

        # error
        predictions = logits.argmax(-1)
        padding_mask = batch['target_padding_mask'][:, 1:].contiguous()

        accuracy = self.accuracy_dev(predictions.view(-1)[padding_mask.view(-1)],
                                     labels.view(-1)[padding_mask.view(-1)])

        self.log('val_accuracy', accuracy, prog_bar=False, on_step=False, on_epoch=True)

        str_input, str_generation, generation = self.generate(batch)

        str_targets = [self.tokenizer.decode(x, skip_special_tokens=True) for x in batch["target"]]


        top_predictions = []
        table = wandb.Table(columns=["Input", "Target", "Merged Generation"])
        for i, (source, predictions, targets) in enumerate(zip(str_input, str_generation, str_targets)):
            merged_predictions = self.merge_return_sequences(predictions)
            top_predictions.append(merged_predictions)
            table.add_data(source, targets, merged_predictions)

        wandb.log({"examples": table})

        prec_at_one = self.val_prec(top_predictions, [x.split(', ') for x in str_targets])
        self.log('val_prec', prec_at_one, prog_bar=False, on_step=False, on_epoch=True)

        self.generated_batches.append((batch["metadata"], str_generation))
        return output['loss'], generation

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, *args, **kwargs):

        str_input, str_generation, generation = self.generate(batch)
        self.generated_batches.append((batch["metadata"], str_generation))

        if self.output_name != '':
            output_folder = self.config["paths"]["output_folder"]
            output_path = os.path.join(output_folder, f"{self.output_name}.txt")

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            if batch_idx == 0:
                print(f'Saving output in {output_path}')
                open_flag = 'w'

            else:
                open_flag = 'a'

            with open(output_path, open_flag) as out:

                for instance, generation in zip(batch["metadata"], str_generation):

                    if instance.gold:

                        sorted_gold = sorted([(k, v) for k, v in instance.gold.items()], key=lambda x: x[1],
                                             reverse=True)
                        sorted_gold_wr = " ".join([f'{x}: {y}' for x, y in sorted_gold])

                    else:
                        sorted_gold_wr = " "

                    out.write(f'{instance.target} {instance.instance_id} '
                              f'{"##".join([str(idx) for idx in instance.target_idx])}\n')
                    out.write(f'{instance.sentence}\n')
                    out.write(f'#gold: {sorted_gold_wr}\n\n')

                    out.write(generation)
                    out.write('\n#########\n')
        return generation


    def get_optimizer_and_scheduler(self):

        no_decay = self.optimiser_dict['no_decay_params']

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.optimiser_dict['weight_decay']
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = RAdam(optimizer_grouped_parameters, self.optimiser_dict['learning_rate'])

        return optimizer, None

    def configure_optimizers(self):

        optimizer, lr_scheduler = self.get_optimizer_and_scheduler()
        if lr_scheduler is None:
            return optimizer
        return [optimizer], [{'interval': 'step', 'scheduler': lr_scheduler}]

    def generate(self, batch: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str], torch.Tensor]:

        generation = self.model.generate(
            input_ids=batch["source"],
            attention_mask=batch["source_padding_mask"],
            **self.generation_parameters,
        )

        generation = generation[:, 1:]

        str_input = [self.tokenizer.decode(x) for x in batch["source"]]
        str_generation = self.handle_beam_generation(generation)
        return str_input, str_generation, generation

    def merge_return_sequences(self, sequence: str) -> str:
        seq = sequence.split(self.split_seq_token)
        words = [w for s in seq for w in s.split(', ')]
        counter = collections.Counter(words)
        most_common = counter.most_common(1)
        return most_common[0][0]

    def handle_beam_generation(self, generation: torch.Tensor) -> List[str]:
        str_generation = []
        for i in range(0, len(generation), self.generation_parameters['num_return_sequences']):
            batch_generation = generation[i: i + self.generation_parameters['num_return_sequences']]
            str_batch = self.split_seq_token.join(
                self.tokenizer.batch_decode(batch_generation, skip_special_tokens=True))
            str_generation.append(str_batch)
        return str_generation
