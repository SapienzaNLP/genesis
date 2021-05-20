import argparse

import tqdm

from src.utils import read_from_raganato_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_path', default='data/semcor/semcor.data.xml')
    parser.add_argument('--keys', default='data/semcor/semcor.gold.key.txt')
    parser.add_argument('--output_path', default='data/semcor_instances.tsv')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:

    with open(args.output_path, 'w') as out:
        for tagged_sentence in tqdm.tqdm(read_from_raganato_format(args.xml_path, args.keys)):
            for instance in tagged_sentence:
                if instance.sense:
                    target = f'{instance.lemma}.{instance.pos}'
                    str_target_idx = '##'.join([str(idx) for idx in instance.target_idx])
                    out.write(f'{target}\t{instance.instance_id}\t{str_target_idx}\t{instance.sentence}\n')


if __name__ == '__main__':
    args = parse_args()
    main(args)
