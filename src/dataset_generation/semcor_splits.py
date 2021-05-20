import argparse
import random

import tqdm

from src.utils import read_from_input_file, file_len

random.seed(42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prev_sample_path', required=False, type=str, default='', help="Path to the biggest SemCor sample already computed (and to include in the current sample)")
    parser.add_argument('--semcor_instances', required=True, type=str, help="Path to the whole semcor instances in the lexsub format")
    parser.add_argument('--k', required=True, type=int, help="Size of the random sample")
    parser.add_argument('--output_path', required=True, type=str, help='Path to the file where the sample will be written')
    return parser.parse_args()


def main(args: argparse.Namespace):
    instances = set([x.instance_id for x in read_from_input_file(args.semcor_instances)])

    if args.prev_sample_path != '':
        exclude_instances = set([x.instance_id for x in read_from_input_file(args.prev_sample_path)])

    else:
        exclude_instances = set()

    remaining = instances.difference(exclude_instances)

    sample = random.sample(remaining, min(args.k - len(exclude_instances), len(remaining)))

    with open(args.output_path, 'w') as out:
        for instance in tqdm.tqdm(read_from_input_file(args.semcor_instances), total=file_len(args.semcor_instances)):
            if instance.instance_id in sample or instance.instance_id in exclude_instances:
                gold = {k: v for k, v in instance.gold.items() if v > 0.7}
                instance.gold = gold
                if len(gold) > 0:
                    out.write(instance.__repr__() + '\n')


if __name__ == '__main__':
    args = parse_args()
    main(args)
