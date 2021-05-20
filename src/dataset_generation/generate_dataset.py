import argparse
import os
from shutil import copyfile

import pytorch_lightning as pl

import yaml
from torch.utils.data import DataLoader

from src.dataset import BartDataset
from src.model import BartModel
from src.utils import yield_batch, read_from_input_file


def get_new_sentences_path(sentences_path_all: str, recovery_path: str):
    processed_instances = set()
    for batch in yield_batch(recovery_path, separator='#########\n'):
        target, instance_id, _ = batch[0].strip().split()
        processed_instances.add(instance_id)

    original_folder, original_name = os.path.split(sentences_path_all)
    new_instances_path = os.path.join(original_folder, original_name.replace('.tsv','_recover_cut.tsv'))
    print(f'Writing new instances file in {new_instances_path}')
    with open(new_instances_path, 'w') as out:
        for instance in read_from_input_file(sentences_path_all):
            if instance.instance_id not in processed_instances:
                out.write(instance.__repr__() + '\n')
    return new_instances_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--beams', type=int, default=0)
    parser.add_argument('--sequences', type=int, default=0)
    parser.add_argument('--cuda_device', type=int)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--sentences_path', required=True)
    parser.add_argument('--recover_from_path', default='')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    cuda_device = args.cuda_device
    configuration = yaml.load(open(args.config_path), Loader=yaml.FullLoader)

    bart_name = configuration['model']['name']
    max_tokens_per_batch = configuration['model']['max_tokens_per_batch']

    if args.recover_from_path == '':
        test_dataset = BartDataset(args.sentences_path, bart_name, max_tokens_per_batch)

    else:
        out_folder, f_name = os.path.split(args.recover_from_path)
        copyfile(args.recover_from_path, os.path.join(out_folder, f_name.replace('.txt', '_recover.txt')))

        new_path = get_new_sentences_path(args.sentences_path, args.recover_from_path)
        test_dataset = BartDataset(new_path, bart_name, max_tokens_per_batch)

    test_dataloader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    if cuda_device == 0:
        model = BartModel.load_from_checkpoint(args.ckpt, strict=False, map_location='cuda')

    else:
        model = BartModel.load_from_checkpoint(args.ckpt, strict=False)

    if model.output_name == '':
        model.output_name = configuration["datasets"]["output_name"]

    model.generation_parameters = configuration["generation_parameters"]

    if args.beams != 0:
        model.generation_parameters["num_beams"] = args.beams

    if args.sequences != 0:
        model.generation_parameters["num_return_sequences"] = args.sequences

    trainer = pl.Trainer(gpus=[cuda_device] if cuda_device is not None else None)

    test_dictionary = trainer.test(test_dataloaders=[test_dataloader], model=model)

    if args.recover_from_path != '':
        out_folder, f_name = os.path.split(args.recover_from_path)
        new_output = os.path.join(out_folder, f_name.replace('.txt', '_new.txt'))
        print(f'Merging output files into {new_output}')

        with open(new_output, 'w') as out:
            for line in open(args.recover_from_path):
                out.write(line)

            for line in open(args.recover_from_path):
                out.write(line)