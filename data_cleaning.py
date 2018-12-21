import pickle
import copy
from search_utils import Voc

data_dir = './data/save/'


#train = pickle.load(open(data_dir + 'small_train_2000000.p', 'rb'))
#valid = pickle.load(open(data_dir + 'small_valid_2000000.p', 'rb'))
train = pickle.load(open(data_dir + 'train_data.p', 'rb'))
valid = pickle.load(open(data_dir + 'test_data.p', 'rb'))
dictionary = pickle.load(open(data_dir + 'whole_data_voc.p', 'rb'))

def clean_voc(dic):

    new_voc = copy.deepcopy(dic)
    index2word = {}
    new_index = 0
    for i in dic.index2word.keys():

        if not dic.index2word[i].isdigit():
            index2word[new_index] = dic.index2word[i]
            new_index += 1

    print('Number of voc in trimmed dictionary {}'.format(len(index2word)))

    word2index = {index2word[index]: index for index in index2word.keys()}

    new_voc.index2word = index2word
    new_voc.word2index = word2index

    return new_voc

new_voc = clean_voc(dictionary)



def eliminate_numeric_sen(SenSet, dic, new_dic):

    modified_voc = copy.deepcopy(dic)

    processed_list = []

    # Eliminate numerical sentences
    for pair_id in range(len(SenSet)):

        for j in range(2):
            breaking = False
            sen = SenSet[pair_id][j]
            for word in sen:
                if dic.index2word[word].isdigit():
                    breaking = True

            if breaking:
                break

            # The second sentence check
            if j == 1:
                temp_list = []
                for sen in SenSet[pair_id]:
                    new_sen = [new_dic.word2index[dic.index2word[index]] for index in sen]
                    temp_list.append(new_sen)

                processed_list.append(temp_list)
    return processed_list

processed_train = eliminate_numeric_sen(train, dictionary, new_voc)
processed_valid = eliminate_numeric_sen(valid, dictionary, new_voc)

#pickle.dump(processed_train, open('processed_train_sen_2000000.p', 'wb'))
#pickle.dump(processed_valid, open('processed_valid_sen_2000000.p', 'wb'))
pickle.dump(processed_train, open('processed_train_sen.p', 'wb'))
pickle.dump(processed_valid, open('processed_valid_sen.p', 'wb'))
pickle.dump(new_voc, open('processed_voc_whole_data.p', 'wb'))


