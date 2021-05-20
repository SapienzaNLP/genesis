import argparse
import os
from shutil import copyfile

import pytorch_lightning as pl

import yaml
from torch.utils.data import DataLoader

from src.dataset import BartDataset
from src.model import BartModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--beams', type=int, default=0)
    parser.add_argument('--sequences', type=int, default=0)
    parser.add_argument('--cuda_device', type=int)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--sentences_path', required=True)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    cuda_device = args.cuda_device
    configuration = yaml.load(open(args.config_path), Loader=yaml.FullLoader)

    bart_name = configuration['model']['name']
    max_tokens_per_batch = configuration['model']['max_tokens_per_batch']

    test_dataset = BartDataset(args.sentences_path, bart_name, max_tokens_per_batch)

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