import argparse
import random
from typing import List, Set

import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig

from src.task_evaluation import get_generated_substitutes, sort_substitutes_cos_sim_batched, is_clean_substitute
from src.utils import LexSubInstance, get_output_dictionary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--cvp', required=True)

    parser.add_argument('--model_name', default="bert-large-cased")
    parser.add_argument('--device', default="cuda")

    parser.add_argument('--threshold', type=float, default=0)
    parser.add_argument('--limit_k_sentences', type=int, default=10000)
    parser.add_argument('--limit_k_substitutes', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=100)
    return parser.parse_args()


def postprocess_generation(generated: List[str]) -> Set[str]:
    words = set()
    for line in generated:
        substitutes = line.strip().split(', ')
        words.update([x.replace(',', '') for x in substitutes])
    return words


def main(args: argparse.Namespace) -> None:

    embedder = AutoModel.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    hs = AutoConfig.from_pretrained(args.model_name).hidden_size

    output_vocabulary = get_output_dictionary(args.cvp)

    instances = []
    instances_infos, input_infos, generated_substitutes, \
    all_generated_substitutes = get_generated_substitutes(args.input_path, output_vocabulary, args.cvp)

    similarities, all_similarities = sort_substitutes_cos_sim_batched(generated_substitutes, input_infos, embedder,
                                                                      tokenizer, hs, args.device, args.threshold,
                                                                      args.batch_size)

    for i, subst_list in tqdm.tqdm(enumerate(generated_substitutes)):

        lexelt, instance_id = instances_infos[i]
        sentence, target_idx = input_infos[i]
        lemma = '.'.join(lexelt.split('.')[:-1])

        if len(similarities[i]) > 0:

            clean_similarities = [(word, score) for word, score in similarities[i]
                                  if (word not in lemma and lemma not in word) and
                                  is_clean_substitute(word)]

            similarities[i] = clean_similarities[:args.limit_k_substitutes]

            instances.append(LexSubInstance(lexelt, instance_id, target_idx, sentence,
                                            gold={w: s for w, s in similarities[i]}))

    random.shuffle(instances)

    target_instances_counter = {}

    with open(args.output_path, 'w') as out:
        for instance in instances:
            if instance.target not in target_instances_counter:
                target_instances_counter[instance.target] = 0

            if len(instance.gold) > 0 and target_instances_counter[instance.target] < args.limit_k_sentences:
                gold_str = " ".join([f'{"_".join(key.split())}::{value}' for key, value in instance.gold.items()])
                mask_str = "-"

                out.write(f'{instance.target}\t{instance.instance_id}\t{instance.target_idx}\t'
                          f'{instance.sentence}\t{mask_str}\t{gold_str}\n')

                target_instances_counter[instance.target] += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
