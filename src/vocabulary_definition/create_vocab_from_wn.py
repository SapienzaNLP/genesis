import os
from typing import List, Set

import nltk
import tqdm
from nltk.corpus import wordnet as wn

from src.utils import read_from_input_file, universal_to_wn_pos


def get_all_related_lemmas(synset: nltk.corpus.reader.wordnet) -> Set[str]:
    related = set()
    related.update(set([l.name() for l in synset.lemmas()]))

    for hyper in synset.hypernyms():
        related.update(set([l.name() for l in hyper.lemmas()]))

    for inst_hyper in synset.instance_hypernyms():
        related.update(set([l.name() for l in inst_hyper.lemmas()]))

    for hypo in synset.hyponyms():
        related.update(set([l.name() for l in hypo.lemmas()]))

    for inst_hypo in synset.instance_hyponyms():
        related.update(set([l.name() for l in inst_hypo.lemmas()]))

    for hol in synset.member_holonyms():
        related.update(set([l.name() for l in hol.lemmas()]))

    for hol in synset.substance_holonyms():
        related.update(set([l.name() for l in hol.lemmas()]))

    for hol in synset.part_holonyms():
        related.update(set([l.name() for l in hol.lemmas()]))

    for mer in synset.member_meronyms():
        related.update(set([l.name() for l in mer.lemmas()]))

    for mer in synset.substance_meronyms():
        related.update(set([l.name() for l in mer.lemmas()]))

    for mer in synset.part_meronyms():
        related.update(set([l.name() for l in mer.lemmas()]))

    for attribute in synset.attributes():
        related.update(set([l.name() for l in attribute.lemmas()]))

    for entailment in synset.entailments():
        related.update(set([l.name() for l in entailment.lemmas()]))

    for cause in synset.causes():
        related.update(set([l.name() for l in cause.lemmas()]))

    for also_see in synset.also_sees():
        related.update(set([l.name() for l in also_see.lemmas()]))

    for verb_group in synset.verb_groups():
        related.update(set([l.name() for l in verb_group.lemmas()]))

    for similar in synset.similar_tos():
        related.update(set([l.name() for l in similar.lemmas()]))

    return related


def get_related_lemmas(lexeme: str) -> Set[str]:
    *lemma, pos = lexeme.split('.')
    lemma = '.'.join(lemma)
    related = set()
    synsets = wn.synsets(lemma, universal_to_wn_pos(pos))

    for synset in synsets:
        # include all neighbours (distance 1)
        related.update(get_all_related_lemmas(synset))

        for also_see in synset.also_sees():
            related.update(get_all_related_lemmas(also_see))

        for similar in synset.similar_tos():
            related.update(get_all_related_lemmas(similar))

        for hypo in synset.hyponyms():
            related.update(get_all_related_lemmas(hypo))

        for hyper in synset.hypernyms():
            related.update(get_all_related_lemmas(hyper))

    return related

def write_to_folders(lexemes_list: List[str], root_output_folder: str) -> None:
    if not os.path.exists(root_output_folder):
        os.makedirs(root_output_folder)

    for lexeme in tqdm.tqdm(lexemes_list):
        substitutes = get_related_lemmas(lexeme)
        with open(os.path.join(root_output_folder, lexeme), 'w') as out:
            for substitute in substitutes:
                out.write(substitute + '\n')


if __name__ == '__main__':
    test_list = [x.target for x in read_from_input_file('data/lst/lst_test.tsv')]
    lexemes_list = list(set(test_list))
    write_to_folders(lexemes_list, 'vocab/wordnet_vocab')
