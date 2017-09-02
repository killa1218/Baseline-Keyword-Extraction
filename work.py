import re
from collections import Counter
import json
import operator
from functools import reduce
import nltk
import networkx as nx
from math import log

def output(big_filename) :
    def toList(sentence):
        grammar = "NP: {<JJS|JJR|JJ|NN|NNS|NNP|NNPS>*<NN|NNS|NNP|NNPS>}"
        cp = nltk.RegexpParser(grammar)
        cp_list = list(cp.parse(sentence))
        tmp_array = []
        for item in cp_list:
            if (type(list(item)[0]) == tuple):
                tmp_string = ''
                for tmp_item in item:
                    if (tmp_string != ''): tmp_string += ' '
                    tmp_string += tmp_item[0]
                tmp_array.append(tmp_string)
            else:
                tmp_array.append(list(item)[0])
        return tmp_array


    def make_np(filename):
        data = json.load(open(filename + '.json', 'r'))
        pos_tag_array = []
        for item in data: pos_tag_array.append(nltk.pos_tag(item['abstract']))

        np = []
        for item in pos_tag_array: np.append(toList(item))

        pt_np = {'pos_tag': pos_tag_array, 'noun_phrase': np}
        json.dump(pt_np, open(filename + '.pt_np', 'w'))


    make_np(big_filename + '/test')
    make_np(big_filename + '/train')
    make_np(big_filename + '/valid')
    print('done ' + big_filename)

def output2(big_filename) :
    def make_key(filename):
        data = json.load(open(filename + '.json', 'r'))

        tmp_array = []
        for item in data:
            ori_array = item['abstract']
            item['keyphrase'].reverse()
            for tmp_item in item['keyphrase']:
                sp = tmp_item['start_position']
                ep = tmp_item['end_position'] + 1
                if (sp + 1 == ep): continue
                combine_str = ''
                for tmp_str in ori_array[sp: ep]:
                    if (combine_str != ''): combine_str += ' '
                    combine_str += tmp_str
                ori_array = ori_array[0: sp] \
                            + [combine_str] \
                            + ori_array[ep: len(ori_array)]
            tmp_array.append(ori_array)

        json.dump({'noun_phrase': tmp_array}, open(filename + '.key_np', 'w'))

    make_key(big_filename + '/test')
    make_key(big_filename + '/train')
    make_key(big_filename + '/valid')
    print('done ' + big_filename)
def count_file(big_filename) :
    def output_count(target) :
        total_counter = Counter()
        total_appear_times = {}
        total_number_of_paper = 0
        data_dict = {'test' : {}, 'train' : {}, 'valid' : {}}
        for name in ['test', 'train', 'valid'] :
            data = json.load(open(big_filename + '/' + name + '.' + target, 'r'))
            total_number_of_paper += len(data['noun_phrase'])
            # count_array = []
            tf_array = []
            for tmp_item in data['noun_phrase']:
                item = [i.lower() for i in tmp_item]
                tmp = Counter(item)
                tmp_dict = dict(sorted(dict(tmp).items(), key=operator.itemgetter(1), reverse=True))

                total_words = reduce(lambda x, value: x + value, tmp_dict.values(), 0)

                for key, value in tmp_dict.items() :
                    if key in total_appear_times :
                        total_appear_times[key] += 1
                    else :
                        total_appear_times[key] = 1

                # count_array.append(tmp_dict)

                tf_array.append(dict([(key, value / total_words) for key, value in tmp_dict.items()]))
                # total_counter += tmp

            data_dict[name]['tf'] = tf_array
            # data_dict[name]['times'] = count_array

        # data_dict['total_counter'] = dict(sorted(dict(total_counter).items(), key=operator.itemgetter(1), reverse=True))
        data_dict_appear_times = dict(sorted(dict(total_appear_times).items(), key=operator.itemgetter(1), reverse=True))

        for name in ['test', 'train', 'valid']:
            # idf_array = []
            tf_idf_array = []
            for i in range(0, len(data_dict[name]['tf'])) :
                paper = data_dict[name]['tf'][i]
                idf_dict = {}
                tf_idf_dict = {}
                for key, tf in paper.items() :
                    idf_dict[key] = log(total_number_of_paper / (data_dict_appear_times[key] + 1))
                    tf_idf_dict[key] = paper[key] * idf_dict[key]

                #idf_dict = dict(sorted(idf_dict.items(), key=operator.itemgetter(1), reverse=True))
                tf_idf_dict = dict(sorted(tf_idf_dict.items(), key=operator.itemgetter(1), reverse=True))
                # data_dict[name]['tf'][i] = dict(list(data_dict[name]['tf'][i]))
                # idf_array.append(idf_dict)
                tf_idf_array.append(tf_idf_dict)


            # data_dict[name]['idf']= idf_array

            data_dict[name]['tf_idf'] = tf_idf_array

        json.dump(data_dict, open(big_filename + '/all_data.' + target, 'w'))
        return data_dict

    def get_output(target, all_data) :
        data_dict = {'test' : [], 'train' : [], 'valid' : []}
        # test : [{n : ?, tf : [], answer : [], tf-idf : []}]

        for name in ['test', 'train', 'valid']:
            data = json.load(open(big_filename + '/' + name + '.json', 'r'))
            for paper in data :
                answer_array = []
                number_of_words = len(paper['keyphrase'])

                for word in paper['keyphrase'] :
                    raw_string = ''
                    sp = word['start_position']
                    ep = word['end_position'] + 1
                    for i in range(sp, ep) :
                        if(raw_string != '') : raw_string += ' '
                        raw_string += paper['abstract'][i]
                    raw_string = raw_string.lower()
                    if(raw_string in answer_array) :
                        number_of_words -= 1
                        continue
                    answer_array.append(raw_string)

                data_dict[name].append({
                    'number_of_word' : number_of_words,
                    'answer' : answer_array
                })

        number_of_limit_type = 4
        limit_type_name = ['3', '5', '10', 'eq']
        total_right = {}
        total_wrong = {}
        total_miss = {}
        for count_method in ['tf', 'tf_idf', 'hits'] :
            total_right[count_method] = [0] * number_of_limit_type
            total_wrong[count_method] = [0] * number_of_limit_type
            total_miss[count_method] = [0] * number_of_limit_type

        for name in ['test', 'train', 'valid']:
            hits_array = json.load(open(big_filename + '/hits_' + name + '_' + target + '.json', 'r'))['hits']
            number_of_paper = len(all_data[name]['tf'])
            for i in range(0, number_of_paper) :
                for count_method in ['tf', 'tf_idf', 'hits'] :
                    if(count_method == 'hits') :
                        tmp_array = sorted(hits_array[i], key=operator.itemgetter(1), reverse=True)
                    else :
                        tmp_array = sorted(list(all_data[name][count_method][i].items()), key=operator.itemgetter(1), reverse=True)
                    #print(tmp_array[0 : 6])
                    number_of_word = data_dict[name][i]['number_of_word']
                    number_of_iter = min(max(10, number_of_word), len(tmp_array))
                    right, wrong = [0] * number_of_limit_type, [0] * number_of_limit_type
                    miss = [0] * number_of_limit_type
                    limit_numbers = [3, 5, 10, number_of_word]

                    for j in range(0, number_of_iter) :

                        if(tmp_array[j][0] in data_dict[name][i]['answer']) :
                            for limit_i in range(0, number_of_limit_type) :
                                if(j < limit_numbers[limit_i]) :
                                    right[limit_i] += 1
                        else :
                            for limit_i in range(0, number_of_limit_type) :
                                if(j < limit_numbers[limit_i]) :
                                    wrong[limit_i] += 1

                    data_dict[name][i][count_method] = {}
                    for limit_i in range(0, number_of_limit_type):
                        miss[limit_i] = number_of_word - right[limit_i]
                        total_right[count_method][limit_i] += right[limit_i]
                        total_wrong[count_method][limit_i] += wrong[limit_i]
                        total_miss[count_method][limit_i] += miss[limit_i]
                        FP = wrong[limit_i]
                        TP = right[limit_i]
                        FN = miss[limit_i]
                        PE = TP * 1.0 / (TP + FP)
                        RE = TP * 1.0 / (TP + FN)
                        if(TP == 0.0) :
                            F1 = 0.0
                        else :
                            F1 = 2 * PE * RE / (PE + RE)
                        data_dict[name][i][count_method][limit_type_name[limit_i]] = \
                            { 'FP' : FP, 'TP' : TP, 'FN' : FN, 'PE' : PE, 'RE' : RE, 'F1' : F1}
        json.dump(data_dict, open(big_filename + '/' + 'result_' + target + '.json', 'w'))

        total_result = {}
        for count_method in ['tf', 'tf_idf', 'hits']:
            total_result[count_method] = {}
            for limit_i in range(0, number_of_limit_type):
                FP = total_wrong[count_method][limit_i]
                TP = total_right[count_method][limit_i]
                FN = total_miss[count_method][limit_i]
                PE = TP * 1.0 / (TP + FP)
                RE = TP * 1.0 / (TP + FN)
                F1 = 2 * PE * RE / (PE + RE)
                total_result[count_method][limit_type_name[limit_i]] = \
                    { 'FP' : FP, 'TP' : TP, 'FN' : FN, 'PE' : PE, 'RE' : RE, 'F1' : F1}

        print(total_result)
        json.dump(total_result, open(big_filename + '/' + 'total_result_' + target + '.json', 'w'))

    tmp_data_dict = output_count('pt_np')
    get_output('pt_np', tmp_data_dict)
    tmp_data_dict = output_count('key_np')
    get_output('key_np', tmp_data_dict)


def new_count_file(big_filename) :

    total_number_of_paper = 0
    total_appear_times = {}
    tf_idf_array = []
    tf_array = {}
    part_data = {}
    for name in ['test', 'train', 'valid'] :
        part_data[name] = json.load(open(big_filename + '/' + name + '.json', 'r'))
        total_number_of_paper += len(part_data[name])
        tf_array[name] = []

        for index, paper in enumerate(part_data[name]):
            paper['abstract'] = [i.lower() for i in paper['abstract']]
            tmp_counter = Counter(paper['abstract'])
            tmp_dict_counter = dict(tmp_counter)
            total_words = reduce(lambda x, value: x + value, tmp_dict_counter.values(), 0)
            for key, value in tmp_dict_counter.items():
                if key in total_appear_times:
                    total_appear_times[key] += 1
                else:
                    total_appear_times[key] = 1

            tf_array[name].append([1.00 * tmp_dict_counter[key] / total_words for key in paper['abstract']])
    total_len = 0
    tmp_tf_idf_array = []

    for name in ['test', 'train', 'valid']:

        for index, paper in enumerate(part_data[name]):
            part_data[name][index]['tf_idf'] = \
                [log(1.00 * total_number_of_paper / (total_appear_times[key] + 1))
                 * tf_array[name][index][tmp_index]
                 for tmp_index, key in enumerate(paper['abstract'])]
            tmp_tf_idf_array.extend(part_data[name][index]['tf_idf'])
            total_len += len(paper['abstract'])

        #json.dump(part_data, open(big_filename + '/' + name + '_tf_idf.json', 'w'))

    boundary = sorted(tmp_tf_idf_array, reverse=True)[100 * total_number_of_paper]

    #print(boundary)

    #print(1.0 * total_len / total_number_of_paper)

    total_number_of_paper = 0
    total_len = 0
    for name in ['test', 'train', 'valid']:
        new_data = []
        for paper in part_data[name] :
            new_abstract = []
            new_tf_idf = []

            dec_key = [0] * len(paper['keyphrase'])
            for index, value in enumerate(paper['tf_idf']) :
                if(value < boundary) :
                    for key_index, keyphrase in enumerate(paper['keyphrase']):
                        if (index < keyphrase['start_position']) :
                            dec_key[key_index] += 1
                            continue
                        elif (index > keyphrase['end_position']) :
                            continue
                        else :
                            dec_key[key_index] = 1000000
                            continue
                else :
                    new_abstract.append(paper['abstract'][index])
                    new_tf_idf.append(value)

            new_phrase = []
            for key_index, keyphrase in enumerate(paper['keyphrase']):
                if(dec_key[key_index] < 1000000) :
                    keyphrase['start_position'] -= dec_key[key_index]
                    keyphrase['end_position'] -= dec_key[key_index]
                    new_phrase.append(keyphrase)

            if(len(new_phrase) > 0) :
                paper['keyphrase'] = new_phrase
                paper['abstract'] = new_abstract
                paper['tf_idf'] = new_tf_idf
                new_data.append(paper)
                total_len += len(new_abstract)

        total_number_of_paper += len(new_data)
        json.dump(new_data, open(big_filename + '/' + name + '_100.json', 'w'))

    print(total_len * 1.0 / total_number_of_paper)

    print('done ' + big_filename)

def make_graph(bigname) :
    for suf in ['key_np', 'pt_np'] :
        for name in ['test', 'train', 'valid'] :
            data_dict = {'hits' : []}
            json_file = open(bigname + '/' + name + '.' + suf, 'r')
            part_data = json.load(json_file)['noun_phrase']
            for data in part_data :

                tmp_size = len(data)
                tmp_dict = {}
                re_dict = []
                tmp_array = []
                tmp_count = 0
                for tmp_word in data :
                    if tmp_word not in tmp_dict :
                        tmp_word = tmp_word.lower()
                        tmp_dict[tmp_word] = tmp_count
                        re_dict.append(tmp_word)
                        tmp_count += 1
                    tmp_array.append(tmp_dict[tmp_word])

                tmp_edge = []
                for i in range(0 ,tmp_count) : tmp_edge.append([])

                for i in range(0, tmp_size) :
                    for j in range(1, 4) :
                        if(i + j >= tmp_size) : continue
                        if(tmp_array[i] == tmp_array[j + i]) : continue
                        tmp_edge[tmp_array[i]].append(tmp_array[i + j])
                        tmp_edge[tmp_array[i + j]].append(tmp_array[i])

                tmp_number_of_iter = 100
                tmp_err_limt = 0.00001
                a_value = [1.0] * tmp_count
                h_value = [1.0] * tmp_count

                for iter_times in range(tmp_number_of_iter) :
                    a_last = a_value
                    a_value = [0.0] * tmp_count
                    for i in range(0, tmp_count) :
                        for j in tmp_edge[i] :
                            a_value[i] += h_value[j]

                    h_value = [0.0] * tmp_count
                    for i in range(0, tmp_count) :
                        for j in tmp_edge[i] :
                            h_value[i] += a_value[j]

                    a_max = max(a_value)
                    h_max = max(h_value)
                    if(a_max == 0) :
                        print(data)
                        break

                    for i in range(0, tmp_count) :
                        a_value[i] /= a_max
                        h_value[i] /= h_max

                    err = sum([abs(a_value[i] - a_last[i]) for i in range(tmp_count)])
                    if(err < tmp_err_limt) : break

                result = sorted([(re_dict[i], a_value[i]) for i in range(tmp_count)], key = lambda x : x[1], reverse=True)
                data_dict['hits'].append(result)

            json.dump(data_dict, open(bigname + '/hits_' + name + '_' + suf + '.json', 'w'))

    print('done ' + bigname)

def make_count_all(big_filename) :

    total_number_of_paper = 0
    total_appear_times = {}
    tf_idf_array = []
    tf_array = {}
    part_data = {}
    for name in ['test', 'train', 'valid'] :
        part_data[name] = json.load(open(big_filename + '/' + name + '.json', 'r'))
        total_number_of_paper += len(part_data[name])
        tf_array[name] = []

        for index, paper in enumerate(part_data[name]):
            paper['abstract'] = [i.lower() for i in paper['abstract']]
            tmp_counter = Counter(paper['abstract'])
            tmp_dict_counter = dict(tmp_counter)
            total_words = reduce(lambda x, value: x + value, tmp_dict_counter.values(), 0)
            for key, value in tmp_dict_counter.items():
                if key in total_appear_times:
                    total_appear_times[key] += 1
                else:
                    total_appear_times[key] = 1

            tf_array[name].append([1.00 * tmp_dict_counter[key] / total_words for key in paper['abstract']])

    for name in ['test', 'train', 'valid']:

        for index, paper in enumerate(part_data[name]):
            part_data[name][index]['tf_idf'] = \
                [log(1.00 * total_number_of_paper / (total_appear_times[key] + 1))
                 * tf_array[name][index][tmp_index]
                 for tmp_index, key in enumerate(paper['abstract'])]

        json.dump(part_data[name], open(big_filename + '/' + name + '_all.json', 'w'))
        print(len(part_data[name][0]['abstract']), len(part_data[name][0]['tf_idf']))


def work(input_func) :
    #input_func('ClearData/krapivin/tokenize/nopunc/nostem')
    #input_func('ClearData/krapivin/tokenize/nopunc/stem')
    input_func('ClearData/ke20k/tokenize/nopunc/nostem')
    input_func('ClearData/ke20k/tokenize/nopunc/stem')

def format_result() :
    def out_put(big_filename) :
        for suf in ['_pt_np', '_key_np'] :
            data = json.load(open(big_filename + '/total_result' + suf + '.json'))
            for method in ['tf_idf', 'tf', 'hits'] :
                for number in ['3', '5', '10', 'eq'] :
                    for gg in ['PE', 'RE', 'F1'] :
                        print(data[method][number][gg], end=' ')
                print(end = '\n')

    out_put('ClearData/krapivin/tokenize/nopunc/nostem')
    out_put('ClearData/krapivin/tokenize/nopunc/stem')
    out_put('ClearData/ke20k/tokenize/nopunc/nostem')
    out_put('ClearData/ke20k/tokenize/nopunc/stem')


if __name__ == "__main__" :
    format_result()
    #work(new_count_file)
    #work(output2)
    #work(make_count_all)
    # work(output_answer)
    # output2('ClearData/krapivin/tokenize/nopunc/nostem')
    # output('ClearData/krapivin/tokenize/nopunc/nostem')
    # count_file('ClearData/krapivin/tokenize/nopunc/nostem')
    # output_answer('ClearData/krapivin/tokenize/nopunc/nostem')
    # output('ClearData/krapivin/tokenize/nopunc/nostem')
    # output('ClearData/krapivin/tokenize/nopunc/stem')
    # output('ClearData/ke20k/tokenize/nopunc/nostem')
    # output('ClearData/ke20k/tokenize/nopunc/stem')