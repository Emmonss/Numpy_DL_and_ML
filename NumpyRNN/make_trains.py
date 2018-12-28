from  data_utils import  extra_input
from RNN_Model import RNN









if __name__ == '__main__':
    file_name = 'input.txt'
    data,data_size, vocab_size, char2index, index2char = extra_input('input.txt', -1)
    model = RNN(vocab_size = vocab_size,
                data_size = data_size,
                data=data,
                char2index = char2index,
                index2char = index2char,
                each_sample_num=1000)

    model.train()