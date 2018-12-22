import torch
import pickle
from search_utils import Voc

#cp = torch.load('./data/save/cb_model/cornell movie-dialogs corpus/2-2_500/best_model_checkpoint.pt')

#train = pickle.load(open('./data/save/small_train_2000000.p', 'rb'))
train = pickle.load(open('../data/save/Generated_data_beam_search_train.p', 'rb'))
valid = pickle.load(open('../data/save/Generated_data_beam_search_valid.p', 'rb'))
#cp = torch.load('/Users/TONY/Downloads/machine_learning/Chatbot_train_dev/data/save/cb_model/cornell movie-dialogs corpus/2-2_500/4000_checkpoint.tar', map_location=lambda storage, loc: storage)
cp = torch.load('/Users/TONY/Downloads/machine_learning/NLP_project/data/save/cb_model/Open_subtitles/3-3_512/best_model_checkpoint_original_setting_no_valid.pt', map_location=lambda storage, loc: storage)
voc = pickle.load(open('/Users/TONY/Downloads/machine_learning/NLP_project/data/save/Vocabulary', 'rb'))
voc.__dict__ = cp['voc_dict']



num_example_sen_pair = 10
print('The top 10 sen in generated train set')
for index in range(num_example_sen_pair):

    for sen in train[index]:
        for word in sen:
            print(voc.index2word[word], end=' ')
        print()

print('The top 10 sen in generated valid set')
for index in range(num_example_sen_pair):

    for sen in valid[index]:
        for word in sen:
            print(voc.index2word[word], end=' ')
        print()


print()

#print(cp['iteration'])
