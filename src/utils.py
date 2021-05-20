import os
import string
import subprocess
from typing import Dict, Optional, List, Iterable, Any, Set

import numpy as np
import xml.etree.ElementTree as ET

import torch
import tqdm
import transformers


_universal_to_lst = {
    'NOUN': 'n',
    'ADJ': 'a',
    'ADV': 'r',
    'VERB': 'v'
}


class LexSubInstance:

    def __init__(self, target: str, instance_id: str, target_idx: List[int], sentence: str,
                 mask: Optional[List[str]] = None, gold: Optional[Dict[str, int]] = None):
        self.target = target
        self.instance_id = instance_id
        self.target_idx = target_idx
        self.sentence = sentence
        self.mask = mask
        self.gold = gold

    def __repr__(self):
        if self.gold:
            sorted_gold = sorted([(a, b) for a, b in self.gold.items()], key=lambda x: x[1], reverse=True)
            f_gold = " ".join([f'{x}::{y}' for x, y in sorted_gold])

            clean_line = '\t'.join([self.target, self.instance_id, str(self.target_idx),
                                    self.sentence, " ".join(self.mask), f_gold])
        else:
            clean_line = '\t'.join([self.target, self.instance_id, str(self.target_idx),
                                    self.sentence])

        return clean_line


def convert_to_universal(pos: str):
    pos = pos.upper()
    if pos in {'NOUN', 'ADJ', 'ADV', 'VERB'}:
        return pos

    if pos.startswith('V'):
        return 'VERB'
    elif pos.startswith('N'):
        return 'NOUN'
    elif pos.startswith('R'):
        return 'ADV'
    elif pos.startswith('J') or pos.startswith('A') or pos.startswith('S'):
        return 'ADJ'
    raise ValueError('Unknown pos tag {}'.format(pos))


def convert_to_universal_target(lexeme: str) -> str:
    *lemma, pos = lexeme.split('.')
    pos = convert_to_universal(pos)
    return ".".join(lemma) + "." + pos


def convert_to_lst_target(target: str) -> str:
    *lemma, pos = target.split('.')
    lemma = '.'.join(lemma)
    pos = _universal_to_lst[pos]
    return f'{lemma}.{pos}'


def read_from_input_file(input_path: str, encoding: str = 'utf-8') -> Iterable[LexSubInstance]:
    for line in open(input_path, encoding=encoding):
        if len(line.strip().split('\t')) == 6:
            target, instance_id, target_idx, sentence, mask, gold = line.strip().split('\t')
            mask = list(set([x for x in mask.split(' ')]))
            gold = {x.split('::')[0]: float(x.split('::')[1]) for x in gold.split(' ')}

        else:
            target, instance_id, target_idx, sentence = line.strip().split('\t')
            mask = None
            gold = None

        *lemma, pos = target.split('.')
        lemma = '.'.join(lemma)
        pos = convert_to_universal(pos)
        target = f'{lemma}.{pos}'
        target_idx = get_target_index_list(target_idx)

        yield LexSubInstance(target=target, instance_id=instance_id, target_idx=target_idx,
                             sentence=sentence, mask=mask, gold=gold)


def flatten(lst: List[list]) -> list:
    return [_e for sub_l in lst for _e in sub_l]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def file_len(file_name):
    return int(subprocess.check_output(f"wc -l {file_name}", shell=True).split()[0])


class WSDInstance:

    def __init__(self, lemma: str, pos: str, word: str, sentence: str, sense: Optional[List[str]] = None,
                 instance_id: Optional[str] = None, target_idx: Optional[int] = None):
        self.instance_id = instance_id
        self.target_idx = target_idx
        self.sentence = sentence
        self.lemma = lemma
        self.pos = pos
        self.word = word
        self.sense = sense

        if ' ' in word:
            len_mw = len(word.split(' '))
            end_idx = target_idx + len_mw -1
            self.target_idx = [target_idx, end_idx]

        else:
            self.target_idx = [self.target_idx]


    def __repr__(self):
        if self.sense:
            return f'{self.sense} {self.instance_id} {self.target_idx} {self.sentence}'

        return f'{self.lemma} {self.pos} {self.sentence}'


def read_from_raganato_format(input_xml: str, input_keys: str) -> Iterable[List[WSDInstance]]:
    sense_keys = {}
    for line in open(input_keys):
        line = line.strip().split(' ')
        key = line[0]
        sense_keys[key] = line[1:]

    root = ET.parse(input_xml).getroot()
    for element in root.iter(tag='corpus'):

        for sentence in element.iter(tag='sentence'):
            sentence_str = ' '.join([x.text for x in sentence.iter() if x.tag in {'wf', 'instance'}])
            instances = []

            instance_index = 0
            for instance in sentence.iter():
                if instance.tag == 'wf':
                    lemma = instance.attrib['lemma']
                    pos = instance.attrib['pos']
                    word = instance.text
                    instances.append(WSDInstance(lemma, pos, word, sentence_str))

                    instance_index += len(word.split(' '))

                elif instance.tag == 'instance':
                    instance_id = instance.attrib['id']
                    lemma = instance.attrib['lemma']
                    pos = instance.attrib['pos']
                    word = instance.text
                    instances.append(WSDInstance(lemma, pos, word, sentence_str, sense_keys[instance_id], instance_id,
                                                 instance_index))

                    instance_index += len(word.split(' '))

            yield instances


def contains(small: List[str], big: List[str]):
    for i in range(len(big) - len(small) + 1):
        for j in range(len(small)):
            if big[i + j] != small[j]:
                break
        else:
            return i, i + len(small)
    return False


def recover_bpes(bpes: List[str], words: List[str], word_idx: int, tokenizer):
    target_word = words[word_idx]
    tokenized = tokenizer.tokenize(target_word)
    if len(tokenized) == 0:
        return None

    start_indexes = [i for i, x in enumerate(bpes) if x == tokenized[0]]

    if not start_indexes:
        return None

    diff = [abs(x - word_idx) for x in start_indexes]

    start_index = np.asarray(start_indexes)[np.argmin(diff)]

    try:
        start, end = contains(tokenized, bpes[start_index:])
        return [x for x in range(start + start_index, end + start_index)]
    except:
        return None


def embed_sentences(embedder: transformers.AutoModel.from_pretrained,
                    tokenizer: transformers.AutoTokenizer.from_pretrained,
                    target_index: [List[List[int]]], sentences: List[str], device: str, hidden_size: int,
                    layer_indexes: List[int], sum: bool = False) -> torch.Tensor:
    idx_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
    matrix = torch.zeros((len(sentences), hidden_size))

    # a list of target indexes for each sentence to embed
    assert len(target_index) == len(sentences)

    with torch.no_grad():
        tokenized = tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding=True, truncation=True)

        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        embedder.to(device)

        hidden_states = embedder(input_ids, attention_mask, output_hidden_states=True)["hidden_states"]

        if not sum:
            hidden_states = torch.mean(torch.stack(hidden_states[layer_indexes[0]:layer_indexes[-1] + 1]), dim=0)
        else:
            hidden_states = torch.sum(torch.stack(hidden_states[layer_indexes[0]:layer_indexes[-1] + 1]), dim=0)

        # batch size x bpes x hidden size
        words = [[x for x in sentence.split(' ') if x != ""] for sentence in sentences]
        bpes = [[idx_to_token[idx.item()] for idx in sentence] for sentence in input_ids]

        for j in range(len(input_ids)):

            target_indexes = target_index[j]

            stacking_vecs = []

            for tix in target_indexes:
                bpes_idx = recover_bpes(bpes[j], words[j], tix, tokenizer)

                if bpes_idx is None:
                    continue

                reconstruct = ''.join(bpes[j][bpes_idx[0]:bpes_idx[-1] + 1]).replace('##', '')
                target = words[j][tix]

                if target != reconstruct:
                    continue

                # 1 x 1 x hidden size
                vecs = torch.mean(hidden_states[j, bpes_idx], dim=0)
                stacking_vecs.append(vecs)

            if stacking_vecs == []:
                print(target_indexes, words[j], 'no targets retrieved')
                continue

            matrix[j] = torch.mean(torch.stack(stacking_vecs), dim=0)

    return matrix


def yield_batch(input_path: str, separator: str):
    lines = []
    for line in open(input_path):
        if line == separator:
            yield lines
            lines = []

        else:
            lines.append(line)

    if lines:
        yield lines


def define_exp_name(config: Dict[str, Any], finetune: bool) -> str:
    exp_name = config['exp_name']
    exp_name = f'{exp_name}_{config["model"]["seed"]}'

    if not finetune:
        exp_name = f'{exp_name}_pt_{config["datasets"]["pretrain"]}'
    else:
        exp_name = f'{exp_name}_pt_{config["datasets"]["pretrain"]}_ft_{config["datasets"]["finetune"]}'


    if 'dropout' in config['model'] and config['model']['dropout'] != 0:
        exp_name = f'{exp_name}_drop_{config["model"]["dropout"]}'

    if 'encoder_layerdropout' in config['model'] and config['model']['encoder_layerdropout']!=0:
        if config['model']['encoder_layerdropout']:
            exp_name = f'{exp_name}_enc_lyd_{config["model"]["encoder_layerdropout"]}'

    if 'decoder_layerdropout' in config['model'] and config['model']['decoder_layerdropout']!=0:
        if config['model']['decoder_layerdropout']:
            exp_name = f'{exp_name}_dec_lyd_{config["model"]["decoder_layerdropout"]}'

    return exp_name


def define_generation_out_folder(config: Dict[str, Any]) -> str:
    shorten_gen_keys = config["shorten_gen_keys"]
    out_name = "_".join(sorted([f'{shorten_gen_keys[k]}_{v}'
                                for k, v in config["generation_parameters"].items()
                                if shorten_gen_keys[k] != None and shorten_gen_keys[k]!='None']))

    return out_name


def contains_punctuation(word: str) -> bool:
    punct = set([x for x in string.punctuation])
    return any(char in punct for char in word)


def get_output_dictionary(output_vocab_folder: str) -> Dict[str, Set[str]]:
    output_vocab = {}
    for filename in tqdm.tqdm(os.listdir(output_vocab_folder)):
        target = filename.replace('.txt', '')
        output_vocab[target] = set()
        for line in open(os.path.join(output_vocab_folder, filename)):
            output_vocab[target].add(line.strip().lower())
    return output_vocab


def get_target_index_list(target_idx: str) -> List[int]:

    if '[' in target_idx:
        target_index = target_idx.replace('[', '').replace(']', '').split(', ')
        target_index = [int(x) for x in target_index]

    elif '##' in target_idx:
        target_index = target_idx.split('##')
        target_index = [int(x) for x in target_index]

    else:
        target_index = [int(target_idx)]

    return target_index


def universal_to_wn_pos(upos: str) -> List[str]:
    pos_map = {'NOUN': ['n'], 'VERB': ['v'], 'ADJ': ['a', 's'], 'ADV': ['r']}
    return pos_map[upos]