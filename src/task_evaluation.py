import argparse
import os
import string
import subprocess
from typing import Dict, List, Set, Tuple, Any, Union, Optional

import numpy as np
import torch
import tqdm
import transformers
import yaml

from src.metrics import precision_at_k, recall_at_k
from src.utils import convert_to_lst_target, embed_sentences, yield_batch, \
    define_generation_out_folder, define_exp_name, contains_punctuation, \
    get_output_dictionary, convert_to_universal_target, get_target_index_list
from src.vocabulary_definition.create_vocab_from_wn import get_related_lemmas


def sort_substitutes_cos_sim_batched(substitutes: List[List[str]],
                                     input_infos: List[Tuple[str, List[int]]],
                                     embedder: transformers.AutoModel.from_pretrained,
                                     tokenizer: transformers.AutoTokenizer.from_pretrained,
                                     hs: int, device: str, threshold: float,
                                     batch_size: int) -> \
        Tuple[List[Union[list, List[Tuple[str, Any]]]], List[List[Tuple[str, Any]]]]:

    stacked_input_sentences, stacked_subst_sentences, stacked_target_indexes, \
    stacked_substitutes_indexes = [], [], [], []

    for i, (input_sentence, target_indexes) in enumerate(input_infos):
        input_words = input_sentence.split()

        for substitute in substitutes[i]:
            substitute_words = input_words[:target_indexes[0]] + [substitute] + input_words[target_indexes[-1] + 1:]
            stacked_subst_sentences.append(" ".join(substitute_words))
            if len(substitute.split()) == 1:
                stacked_substitutes_indexes.append([target_indexes[0]])
            else:
                stacked_substitutes_indexes.append([target_indexes[0],
                                                   target_indexes[0] + len(substitute.split()) - 1])

        stacked_input_sentences.append(input_sentence)
        stacked_target_indexes.append(target_indexes)

    input_matrix_embed = torch.zeros((len(stacked_input_sentences), hs), device=device)

    for i in tqdm.tqdm(range(0, len(stacked_input_sentences), batch_size), desc='Embedding input sentences'):
        batch = stacked_input_sentences[i: i + batch_size]
        batch_indexes = stacked_target_indexes[i: i + batch_size]
        vecs = embed_sentences(embedder, tokenizer, batch_indexes, batch, device, hidden_size=hs,
                               layer_indexes=[20, 23])
        input_matrix_embed[i: i + batch_size] = vecs

    subst_matrix_embed = torch.zeros((len(stacked_subst_sentences), hs), device=device)

    for i in tqdm.tqdm(range(0, len(stacked_subst_sentences), batch_size), desc='Embedding substitutes'):
        batch_subst = stacked_subst_sentences[i: i + batch_size]
        batch_indexes_subst = stacked_substitutes_indexes[i: i + batch_size]

        vecs = embed_sentences(embedder, tokenizer, batch_indexes_subst, batch_subst, device, hidden_size=hs,
                               layer_indexes=[20, 23])

        subst_matrix_embed[i: i + batch_size] = vecs

    similarities, all_similarities_list = [], []

    start_idx = 0
    for j, element in enumerate(substitutes):

        input_vec = input_matrix_embed[j]

        if len(element) > 0:
            subst_vec = subst_matrix_embed[start_idx: start_idx + len(element)]

        else:
            # i don't have input substitutes to rank
            similarities.append([])
            all_similarities_list.append([])
            continue

        cos_similarities = torch.nn.functional.cosine_similarity(subst_vec, input_vec.repeat(subst_vec.shape[0], 1))

        paired_similarities = [(substitute, s.item()) for substitute, s in zip(element, cos_similarities)
                               if s.item() >= threshold]
        all_similarities = [(substitute, s.item()) for substitute, s in zip(element, cos_similarities)]

        sorted_substitutes = sorted(paired_similarities, key=lambda x: x[1], reverse=True)
        sorted_all_substitutes = sorted(all_similarities, key=lambda x: x[1], reverse=True)

        similarities.append(sorted_substitutes)
        all_similarities_list.append(sorted_all_substitutes)

        start_idx += len(element)

    return similarities, all_similarities_list


def get_clean_substitutes_from_batch(batch: List[str], target: str,
                                     whole_vocab: Optional[Set[str]] = None,
                                     output_vocabulary: Optional[Union[Set[str],
                                                                       Dict[str, Set[str]]]] = None,
                                     root_vocab_path: Optional[str] = None) -> Tuple[
    List[str], List[str]]:

    substitutes = [[y.strip(',').lower() for y in x.strip().split(', ')] for x in batch]

    clean_substitutes = []
    all_substitutes = []

    *lemma_target, pos_target = target.split('.')
    lemma_target = '.'.join(lemma_target)

    for sub_line in substitutes:
        for word in sub_line:
            if word not in all_substitutes:
                if check_word(word, lemma_target):
                    if whole_vocab:
                        if word in whole_vocab:
                            all_substitutes.append(word)
                    else:
                        all_substitutes.append(word)

            if word not in clean_substitutes:
                if output_vocabulary:
                    if isinstance(output_vocabulary, Set):
                        if word not in output_vocabulary:
                            continue
                    else:
                        assert isinstance(output_vocabulary, Dict)
                        if target in output_vocabulary:
                            if word.lower() not in output_vocabulary[target]:
                                continue
                        else:
                            substitutes = get_related_lemmas(target)

                            output_vocabulary[target] = substitutes
                            with open(os.path.join(root_vocab_path, target), 'w') as out:
                                for substitute in substitutes:
                                    out.write(substitute + '\n')

                if check_word(word, lemma_target):
                    if whole_vocab:
                        if word in whole_vocab:
                            clean_substitutes.append(word)
                    else:
                        clean_substitutes.append(word)

    return clean_substitutes, all_substitutes

def is_clean_substitute(word: str) -> bool:
    return not contains_punctuation(word)

def check_word(word: str, lemma: str) -> bool:

    punctuation = set([x for x in string.punctuation])
    if word == '':
        return False

    if word[-1] in punctuation:
        return False

    if '-' in word:
        return False

    if len(word.split()) > 1:
        return False

    if word not in lemma and not lemma in word:
        return True

    return False


def get_sorted_substitutes_from_batch(batch: List[str], lemma_target: str) -> List[str]:

    substitutes = [[y.strip(',').lower() for y in x.strip().split(', ')] for x in batch]
    no_tar_substitutes = [[x for x in list_subst if x != lemma_target] for list_subst in substitutes]

    counter = {}
    for sequence in no_tar_substitutes:
        for i, word in enumerate(sequence):
            if word not in counter:
                counter[word] = []
            counter[word].append(i + 1)

    for word in counter:
        counter[word] = np.mean(counter[word])

    sorted_substitutes = sorted([(k, v) for k, v in counter.items()], key=lambda x: x[1])
    sorted_substitutes = [word for word, score in sorted_substitutes]
    return sorted_substitutes


def compute_baseline(dataset_name: str, input_path: str, output_folder: str,
                    gold_dict_per_instance: Dict,
                    model_name: str, device: str, threshold: float,
                    top_k: int = 10, batch_size: int = 100,
                    output_vocabulary: Optional[Union[Set[str], Dict[str, Set[str]]]]=None):

    oot_output_path = os.path.join(output_folder, f'{dataset_name}_baseline_oot.txt')
    best_output_path = os.path.join(output_folder, f'{dataset_name}_baseline_best.txt')
    output_path = os.path.join(output_folder, f'{dataset_name}_baseline_hr_output.txt')

    instances_infos, input_infos, generated_substitutes, all_gen_subst = get_generated_substitutes(input_path)

    embedder = transformers.AutoModel.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    hidden_size = transformers.AutoConfig.from_pretrained(model_name).hidden_size

    for s_idx in range(len(all_gen_subst)):
        target = convert_to_universal_target(instances_infos[s_idx][0])
        all_gen_subst[s_idx] = list(output_vocabulary[target])


    similarities, all_similarities = sort_substitutes_cos_sim_batched(all_gen_subst, input_infos,
                                                                      embedder, tokenizer, hidden_size,
                                                                      device, threshold, batch_size)

    with open(best_output_path, 'w', encoding='utf-8') as best, open(oot_output_path, 'w', encoding='utf-8') as oot, \
            open(output_path, 'w', encoding='utf-8') as readable:

        avg_prec_1, avg_prec_3 = 0, 0
        avg_recall_10 = 0
        tot = 0

        for i, subst_list in tqdm.tqdm(enumerate(generated_substitutes), total=len(generated_substitutes)):

            all_sorted_substitutes_cos_sim = [word for word, score in similarities[i]]

            lexelt, instance_id = instances_infos[i]
            key = f'{lexelt} {instance_id}'

            tot += 1

            avg_recall_10 += recall_at_k(all_sorted_substitutes_cos_sim, gold_dict_per_instance[key], k=10)
            oot.write(f'{lexelt} {instance_id} ::: {";".join(all_sorted_substitutes_cos_sim[:top_k])}\n')

            avg_prec_1 += precision_at_k(all_sorted_substitutes_cos_sim, gold_dict_per_instance[key], k=1)
            avg_prec_3 += precision_at_k(all_sorted_substitutes_cos_sim, gold_dict_per_instance[key], k=3)

            if len(all_sorted_substitutes_cos_sim) > 0:
                best.write(f'{lexelt} {instance_id} :: {all_sorted_substitutes_cos_sim[0]}\n')

            count_str_cos = '; '.join([f"{k}: {np.round(v, 2)}" for k, v in similarities[i]
                                       if k in generated_substitutes[i]])

            gener_str_cos = '; '.join(list(all_gen_subst[i]))

            input_sentence, target_index = input_infos[i]

            if any(gen in all_gen_subst[i] and not gen in generated_substitutes[i]
                   for gen in gold_dict_per_instance[key]):
                readable.write(f'@@@ ')

            readable.write(f'{key} {target_index}\n'
                           f'#input: {input_sentence}\n'
                           f'#gold: {" ".join(gold_dict_per_instance[key])}\n'
                           f'#generated: {gener_str_cos}\n'
                           f'#clean: {count_str_cos}\n\n')

    print(f'Precision@1: {np.round(avg_prec_1 / tot, 3) * 100}\n'
          f'Precision@3: {np.round(avg_prec_3 / tot, 3) * 100}\n'
          f'Recall@10: {np.round(avg_recall_10 / tot, 3) * 100}\n'
          f'Retrieved instances: {tot}')

    return best_output_path, oot_output_path


def eval_generation(dataset_name: str, input_path: str, output_folder: str,
                    gold_dict_per_instance: Dict,
                    model_name: str, device: str, threshold: float, suffix: str, backoff: bool,
                    root_vocab_path: str,
                    top_k: int = 10, batch_size: int = 100, baseline: bool = False,
                    cut_vocab: bool = False,
                    output_vocabulary: Optional[Union[Set[str], Dict[str, Set[str]]]]=None) -> Tuple[str, str]:

    embedder = transformers.AutoModel.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    hidden_size = transformers.AutoConfig.from_pretrained(model_name).hidden_size

    if baseline:
        return compute_baseline(dataset_name, input_path, output_folder, gold_dict_per_instance, model_name, device,
                                threshold, top_k, batch_size, output_vocabulary)

    if cut_vocab:

        oot_output_path = os.path.join(output_folder, f'{dataset_name}_cut_per_target_{suffix}_oot.txt')
        best_output_path = os.path.join(output_folder, f'{dataset_name}_cut_per_target_{suffix}_best.txt')
        output_path = os.path.join(output_folder, f'{dataset_name}_cut_per_target_hr_{suffix}_output.txt')

    else:
        oot_output_path = os.path.join(output_folder, f'{suffix}_{dataset_name}_oot.txt')
        best_output_path = os.path.join(output_folder, f'{suffix}_{dataset_name}_best.txt')
        output_path = os.path.join(output_folder, f'{suffix}_{dataset_name}_hr_output.txt')

    print(f'Saving readable output in {output_path}')

    gold_per_target = {}
    for key, value in gold_dict_per_instance.items():
        lexelt, instance_id = key.split()
        if lexelt not in gold_per_target:
            gold_per_target[lexelt] = set()
        gold_per_target[lexelt].update(value)


    with open(best_output_path, 'w', encoding='utf-8') as best, open(oot_output_path, 'w', encoding='utf-8') as oot, \
            open(output_path, 'w', encoding='utf-8') as readable:

        avg_prec_1, avg_prec_3 = 0, 0
        avg_recall_10 = 0
        tot = 0

        if cut_vocab:
            instances_infos, input_infos, \
            generated_substitutes, all_gen_subst = get_generated_substitutes(input_path, output_vocabulary, root_vocab_path)

        else:
            instances_infos, input_infos, generated_substitutes, all_gen_subst = get_generated_substitutes(input_path)


        backoff_list = []
        for s_idx in range(len(all_gen_subst)):
            target = convert_to_universal_target(instances_infos[s_idx][0])
            if backoff:
                backoff_list.append(list(output_vocabulary[target]))

        similarities, _ = sort_substitutes_cos_sim_batched(all_gen_subst, input_infos,
                                                                          embedder, tokenizer, hidden_size,
                                                                          device, threshold, batch_size)

        if backoff:
            # sort by cos sim with the target all the possible substitutes in the output vocab
            backoff_sorted, _ = sort_substitutes_cos_sim_batched(backoff_list, input_infos,
                                                                 embedder, tokenizer, hidden_size,
                                                                 device, threshold, batch_size)

        for i, subst_list in tqdm.tqdm(enumerate(generated_substitutes), total=len(generated_substitutes)):

            clean_sorted_substitutes = [word for word, score in similarities[i] if word in generated_substitutes[i]]
            all_sorted_substitutes_cos_sim = [word for word, score in similarities[i]]

            lexelt, instance_id = instances_infos[i]
            key = f'{lexelt} {instance_id}'

            tot += 1

            if len(clean_sorted_substitutes) < top_k and backoff:

                compound_list = all_sorted_substitutes_cos_sim + [word for word, score in backoff_sorted[i]
                                                                  if word not in all_sorted_substitutes_cos_sim]

                compound_list = [x.replace('_', ' ') for x in compound_list]
                to_wr_subst = [(word, score) for word, score  in similarities[i]
                               if word in generated_substitutes[i]] + \
                              [(word, score) for word, score in backoff_sorted[i]
                               if word not in clean_sorted_substitutes]

            else:
                compound_list = clean_sorted_substitutes
                to_wr_subst = [(word, score) for word, score  in similarities[i]
                               if word in generated_substitutes[i]]


            avg_recall_10 += recall_at_k(compound_list, gold_dict_per_instance[key], k=10)
            oot.write(f'{lexelt} {instance_id} ::: {";".join(compound_list[:top_k])}\n')

            avg_prec_1 += precision_at_k(clean_sorted_substitutes, gold_dict_per_instance[key], k=1)
            avg_prec_3 += precision_at_k(clean_sorted_substitutes, gold_dict_per_instance[key], k=3)

            if len(clean_sorted_substitutes) > 0:
                best.write(f'{lexelt} {instance_id} :: {clean_sorted_substitutes[0]}\n')

            count_str_cos = '; '.join([f"{k}: {np.round(v, 2)}" for k, v in to_wr_subst])

            gener_str_cos = '; '.join(list(all_gen_subst[i]))

            input_sentence, target_index = input_infos[i]

            if any(gen in all_gen_subst[i] and not gen in generated_substitutes[i]
                   for gen in gold_dict_per_instance[key]):
                readable.write(f'@@@ ')

            readable.write(f'{key} {target_index}\n'
                           f'#input: {input_sentence}\n'
                           f'#gold: {" ".join(gold_dict_per_instance[key])}\n'
                           f'#generated: {gener_str_cos}\n'
                           f'#clean: {count_str_cos}\n\n')

    print(f'Precision@1: {np.round(avg_prec_1 / tot, 3) * 100}\n'
          f'Precision@3: {np.round(avg_prec_3 / tot, 3) * 100}\n'
          f'Recall@10: {np.round(avg_recall_10 / tot, 3) * 100}\n'
          f'Retrieved instances: {tot}')

    return best_output_path, oot_output_path


def get_generated_substitutes(input_path: str,
                              output_vocabulary: Optional[Union[Set[str], Dict[str, Set[str]]]]=None,
                              root_vocab_path: Optional[str] = None):


    instances_infos, input_infos, generated_substitutes, all_generated_substitutes = [], [], [], []

    for sentences in yield_batch(input_path, separator='#########\n'):
        instance, instance_id, *target_index = sentences[0].strip().split(' ')
        target_index = ' '.join(target_index)

        input_sentence = sentences[1].strip()

        target_index = get_target_index_list(target_index)
        lexelt = convert_to_lst_target(instance)

        # group substitutes generated across different beams, remove target word and substitutes not in cut vocab
        if output_vocabulary:

            clean_substitutes, all_substitutes = get_clean_substitutes_from_batch(sentences[4:], instance,
                                                                                  output_vocabulary=output_vocabulary,
                                                                                  root_vocab_path=root_vocab_path
                                                                                  )

        else:
            clean_substitutes, all_substitutes = get_clean_substitutes_from_batch(sentences[4:], instance)

        input_infos.append((input_sentence, target_index))
        generated_substitutes.append(clean_substitutes)
        all_generated_substitutes.append(all_substitutes)

        instances_infos.append((lexelt, instance_id))

    return instances_infos, input_infos, generated_substitutes, all_generated_substitutes


def eval_on_task(config: Dict, best_path: str, oot_path: str, dataset_name: str) -> None:

    gold_path = os.path.join(config['paths']['scorer_dir'], f'{dataset_name}_gold.txt')

    # best eval on cosine similarity
    print('\nSorting by cos-sim')
    scorer_path = os.path.join(config['paths']['scorer_dir'], 'score.pl')
    command = ['perl', scorer_path, best_path, gold_path]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print(result.stdout)

    # oot eval
    command = ['perl', scorer_path, oot_path, gold_path, '-t', 'oot']
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print(result.stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--suffix', required=True)
    parser.add_argument('--cvp', required=True)

    parser.add_argument('--beams', type=int, default=15)
    parser.add_argument('--sequences', type=int, default=3)

    parser.add_argument('--embedder', default="bert-large-cased")
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--baseline', default=False, action="store_true")
    parser.add_argument('--backoff', default=False, action="store_true")
    parser.add_argument('--cut_vocab', default=False, action="store_true")
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--finetune', default=False, action="store_true")
    parser.add_argument('--test', default=False, action="store_true")

    return parser.parse_args()


def main(args: argparse.Namespace):

    config = yaml.load(open(args.config_path), Loader=yaml.FullLoader)
    map_location = 'cuda' if args.cuda_device == 0 else 'cpu'

    if args.beams != 0:
        config["generation_parameters"]["num_beams"] = args.beams

    if args.sequences != 0:
        config["generation_parameters"]["num_return_sequences"] = args.sequences


    if args.seed != 0:
        config["model"]["seed"] = args.seed

    if args.test:
        dataset_name = config['datasets']['test']

    else:
        print(f'Testing on dev set.')
        dataset_name = config['datasets']['dev']

    out_name = define_generation_out_folder(config)

    exp_name = define_exp_name(config, finetune=args.finetune)

    input_folder = os.path.join(config['paths']['output_folder'], exp_name, out_name)
    output_folder = os.path.join(input_folder, 'output_files')
    input_path = os.path.join(output_folder, f'output_{dataset_name}.txt')

    print(f'Computing results for beam size {config["generation_parameters"]["num_beams"]}, '
          f'{config["generation_parameters"]["num_return_sequences"]} returned sequences.')
    print(f'Reading predictions from {input_path}')

    if args.cut_vocab:
        cut_vocab_path = args.cvp
        output_vocabulary = get_output_dictionary(cut_vocab_path)

    else:
        output_vocabulary = set()

    gold_path = os.path.join(config['paths']['scorer_dir'], f'{dataset_name}_gold.txt')
    gold_per_instance = {}

    for line in open(gold_path):
        lex, candidates = line.strip().split(' :: ')
        gold_substitutes = [' '.join(x.split(' ')[:-1]) for x in candidates.split(';') if x != '']
        gold_per_instance[lex] = gold_substitutes

    best_path, oot_path = eval_generation(dataset_name, input_path, output_folder,
                                          output_vocabulary=output_vocabulary,
                                          root_vocab_path=args.cvp,
                                          gold_dict_per_instance=gold_per_instance,
                                          model_name=args.embedder, device=map_location,
                                          suffix=args.suffix,
                                          baseline=args.baseline,
                                          backoff=args.backoff,
                                          cut_vocab=args.cut_vocab)

    eval_on_task(config, best_path, oot_path, dataset_name)

if __name__ == '__main__':
    args = parse_args()
    main(args)
