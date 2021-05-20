'''
This code is adapted from https://github.com/orenmel/lexsub
Used to compute GAP score for the LST ranking task

'''

import sys
import random
import re

from scoring_scripts.generalized_average_precision import GeneralizedAveragePrecision


#take.v 25 :: consider 2;accept 1;include 1;think about 1;
def read_gold_line(gold_line, ignore_mwe):
    segments = gold_line.split("::")
    instance_id = segments[0].strip()
    gold_weights = []
    line_candidates = segments[1].strip().split(';')
    for candidate_count in line_candidates:
        if len(candidate_count) > 0:
            delimiter_ind = candidate_count.rfind(' ')
            candidate = candidate_count[:delimiter_ind]
            if ignore_mwe and (len(candidate.split(' ')) > 1 or len(candidate.split('-')) > 1):
                continue
            count = candidate_count[delimiter_ind:]
            try:
                gold_weights.append((candidate, int(count)))
            except ValueError as e:
                print(e)
                print(gold_line)
                print("cand=%s count=%s" % (candidate,count))
                sys.exit(1)
        
    return instance_id, gold_weights

#RESULT    find.v 71    show 0.34657
def read_eval_line(eval_line):
    eval_weights = []
    segments = eval_line.split("\t")
    instance_id = segments[1].strip()
    for candidate_weight in segments[2:]:
        if len(candidate_weight) > 0:
            delimiter_ind = candidate_weight.rfind(' ')
            candidate = candidate_weight[:delimiter_ind]
            weight = candidate_weight[delimiter_ind:]
            if ignore_mwe and ((len(candidate.split(' '))>1) or (len(candidate.split('-'))>1)):
                continue
            try:
                eval_weights.append((candidate, float(weight)))
            except:
                if candidate!='' and candidate!='\n':
                    print("Error appending: %s %s" % (candidate, weight))

    return instance_id, eval_weights
    

if __name__ == '__main__':
    
    if len(sys.argv) < 4:
        print("usage: %s <gold-filename> <eval-filename> <output-filename> [no-mwe] [random]"  % (sys.argv[0]))
        sys.exit(1)
        
    gold_file = open(sys.argv[1], 'r')
    eval_file = open(sys.argv[2], 'r')
    out_file = open(sys.argv[3], 'w')
    
    if len(sys.argv) > 4 and sys.argv[4] == 'no-mwe':
        ignore_mwe = True
    else:
        ignore_mwe = False
        
    if len(sys.argv) > 5 and sys.argv[5] == 'random':
        randomize = True
    else:
        randomize = False
    
    gold_data = {}
    eval_data = {}
    
    i=0
    sum_gap = 0.0
    for eval_line in eval_file:
        eval_instance_id, eval_weights = read_eval_line(eval_line)
        eval_data[eval_instance_id] = eval_weights

    for gold_line in gold_file:
        gold_instance_id, gold_weights = read_gold_line(gold_line, ignore_mwe)
        gold_data[gold_instance_id] = gold_weights
        
    ignored = 0
    for gold_instance_id, gold_weights in gold_data.items():
        if gold_instance_id in eval_data:
            eval_weights = eval_data[gold_instance_id]
            gap = GeneralizedAveragePrecision.calc(gold_weights, eval_weights, randomize)
            if (gap < 0): # this happens when there is nothing left to rank after filtering the multi-word expressions
                ignored += 1
                continue
            out_file.write(gold_instance_id + "\t" + str(gap) + "\n")
            i += 1
            sum_gap += gap
        else:
            print(f'instance {gold_instance_id} not in eval data. Skipping instance.')

    mean_gap = sum_gap/i
    out_file.write("\ngold_data %d eval_data %d\n" % (len(gold_data),len(eval_data)))
    out_file.write("\nRead %d test instances\n" % i)
    out_file.write("\nIgnored %d test instances (couldn't compute gap)\n" % ignored)
    out_file.write("\nMEAN_GAP\t" + str(mean_gap) + "\n")
    
     
    gold_file.close()
    eval_file.close()
    out_file.close()