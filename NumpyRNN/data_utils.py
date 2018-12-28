import numpy as np


def extra_input(filename,length):
    data = open(filename,'r').read()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    char2index = {ch:i for i,ch in enumerate(chars) }
    index2char = {i:ch for i,ch in enumerate(chars) }
    return data,data_size,vocab_size,char2index,index2char






if __name__ == '__main__':
    data,data_size, vocab_size, char2index, index2char = extra_input('input.txt',-1)
    print(data_size)
    print(vocab_size)
    print(char2index)
    print(index2char)
    # inputs, p = batch_input_one_hot_seq(input, p, SEQ_SIZE, BATCH_SIZE, VOCABULARY_SIZE, char_to_index)
    # print(inputs)



