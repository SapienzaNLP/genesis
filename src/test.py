import argparse
import os

import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import BartDataset
from src.model import BartModel
from src.task_evaluation import eval_generation, eval_on_task
from src.utils import define_generation_out_folder, get_output_dictionary


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', required=True, help='path to the yaml config file')

    parser.add_argument('--cuda_device', type=int, default=None)

    parser.add_argument('--suffix', required=True, help='name to be used as suffix for saving output files')

    parser.add_argument('--cvp', required=True,
                        help='path to the output vocabulary. If "", the vocabulary won\'t be cut')

    parser.add_argument('--ckpt', required=True, help='path to the checkpoint to test.')

    parser.add_argument('--beams', type=int, default=15, help='beam size for beam search during evaluation.')
    parser.add_argument('--sequences', type=int, default=3,
                        help='number of top k sequences generated with beam search to consider during evaluation.')

    parser.add_argument('--threshold', type=float, default=0,
                        help='threshold for cosine similarity between candidate substitute and target word')

    parser.add_argument('--embedder', default="bert-large-cased",
                        help='embedder used to compute contextualised representations.')

    parser.add_argument('--cut_vocab', default=False, action="store_true",
                        help='flag. If set, will cut over the output vocabulary specified with --cvp')

    parser.add_argument('--baseline', default=False, action="store_true",
                        help='flag. If set, will compute the vocabulary baseline')

    parser.add_argument('--backoff', default=False, action="store_true",
                        help='flag. If set, will use the fallback strategy')

    parser.add_argument('--test', default=False, action="store_true",
                        help='flag. If set, will evaluate on the test dataset instead of the dev one.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    configuration = yaml.load(open(args.config_path), Loader=yaml.FullLoader)

    bart_name = configuration['model']['name']
    max_tokens_per_batch = configuration['model']['max_tokens_per_batch']

    data_dir = configuration['paths']['data_dir']

    if args.test:
        dataset_name = configuration['datasets']['test']

    else:
        print(f'Testing on dev')
        dataset_name = configuration['datasets']['dev']

    map_location = 'cuda' if args.cuda_device == 0 else 'cpu'

    if args.test:
        test_path = os.path.join(data_dir, dataset_name, f'{dataset_name}_test.tsv')

    else:
        test_path = os.path.join(data_dir, dataset_name, f'{dataset_name}_dev.tsv')

    test_dataset = BartDataset(test_path, bart_name, max_tokens_per_batch)

    model = BartModel.load_from_checkpoint(args.ckpt, strict=False, map_location=map_location)

    test_dataloader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    model.generation_parameters = configuration["generation_parameters"]

    if args.beams != 0:
        model.generation_parameters["num_beams"] = args.beams

    if args.sequences != 0:
        model.generation_parameters["num_return_sequences"] = args.sequences

    trainer = pl.Trainer()

    test_dictionary = trainer.test(test_dataloaders=[test_dataloader], model=model)

    out_name = define_generation_out_folder(configuration)

    ckpt_path, ckpt_name = os.path.split(args.ckpt)
    input_folder = os.path.split(os.path.split(ckpt_path)[0])[0]
    output_folder = os.path.join(input_folder, out_name, 'output_files')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, f'output_{args.suffix}_{dataset_name}.txt')

    print(f'Saving output in {output_path}')

    with open(output_path, 'w') as out:

        for idx, element in enumerate(model.generated_batches):
            instance_batch, generated_batch = element

            for instance, generation in zip(instance_batch, generated_batch):
                sorted_gold = sorted([(k, v) for k,v in instance.gold.items()], key=lambda x:x[1], reverse=True)
                sorted_gold_wr = " ".join([f'{x}: {y}' for x, y in sorted_gold])

                out.write(f'{instance.target} {instance.instance_id} {instance.target_idx}\n')
                out.write(f'{instance.sentence}\n')
                out.write(f'#gold: {sorted_gold_wr}\n\n')

                out.write(generation)
                out.write('\n#########\n')

    if args.cut_vocab:
        output_vocab_path = args.cvp
        output_vocabulary = get_output_dictionary(output_vocab_path)

    else:
        output_vocabulary = set()

    if args.test:
        gold_path = os.path.join(configuration['paths']['scorer_dir'], f'{dataset_name}_gold.txt')

    else:
        gold_path = os.path.join(configuration['paths']['scorer_dir'], f'{dataset_name}_dev_gold.txt')

    gold_per_instance = {}

    for line in open(gold_path):
        lex, candidates = line.strip().split(' :: ')
        gold_substitutes = [' '.join(x.split(' ')[:-1]) for x in candidates.split(';') if x != '']
        gold_per_instance[lex] = gold_substitutes


    if torch.cuda.is_available():
        map_location = 'cuda'

    else:
        map_location = 'cpu'

    best_path, oot_path = eval_generation(dataset_name, output_path, output_folder,
                                          output_vocabulary=output_vocabulary,
                                          gold_dict_per_instance=gold_per_instance,
                                          model_name=args.embedder, device=map_location,
                                          threshold=args.threshold,
                                          backoff=args.backoff,
                                          suffix=args.suffix,
                                          baseline=args.baseline,
                                          cut_vocab=args.cut_vocab,
                                          root_vocab_path=args.cvp)

    eval_on_task(configuration, best_path, oot_path, dataset_name)
