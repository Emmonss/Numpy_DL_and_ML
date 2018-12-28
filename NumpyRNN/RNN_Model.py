import numpy as np


class RNN:
    def __init__(self,
                 vocab_size,
                 char2index,
                 index2char,
                 data,
                 data_size,
                 each_sample_num=100,
                 hidden_size=100,
                 seq_length = 25,
                 lr = 0.01,
                 epoch = 100000
                 ):
        self.hidden_size = hidden_size
        self.lr = lr
        self.epoch = epoch
        self.vocab_size = vocab_size
        self.data = data
        self.data_size = data_size
        self.char2index = char2index
        self.index2char = index2char
        self.seq_length = seq_length
        self._init_characters_()

    def _init_characters_(self):
        self.Wxh = np.random.randn(self.hidden_size,self.vocab_size)
        self.Whh = np.random.randn(self.hidden_size,self.hidden_size)
        self.Why = np.random.randn(self.vocab_size,self.hidden_size)
        self.bh = np.zeros((self.hidden_size,1))
        self.by = np.zeros((self.vocab_size,1))


    def Compute_LossFunction(self,inputs,targets,hprev):
        '''
        :param inputs:
        :param targets:
        :param hprev:
        :return: loss,gradients
        '''
        xs, hs, ys, ps = {},{},{},{}
        hs[-1] = np.copy(hprev)
        loss = 0

        #forward
        for t in range(len(targets)):
            #one-hot
            xs[t] = np.zeros((self.vocab_size,1))
            xs[t][inputs[t]] = 1

            #hidden state
            hs[t] = np.tanh(np.dot(self.Wxh,xs[t])+np.dot(self.Whh,hs[t-1])+self.bh)
            #下一个词的概率
            ys[t] = np.dot(self.Why,hs[t]) + self.by
            #softmax
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            #loss
            loss += -np.log(ps[t][targets[t],0])

        #计算梯度
        dWxh,dWhh,dWhy = np.zeros_like(self.Wxh),np.zeros_like(self.Whh),np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh),np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])


        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= -1


            dWhy += np.dot(dy,hs[t].T)
            dby += dy


            dh = np.dot(self.Why.T, dy) + dhnext
            dhraw = (1 - hs[t]*hs[t]) * dh
            dbh += dhraw


            dWxh += np.dot(dhraw , xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)

            dhnext = np.dot(self.Whh.T,dhraw)

        for dparam in [dWxh,dWhh,dWhy,dbh,dby]:
            np.clip(dparam, -5,5,out = dparam)

        return loss,dWxh,dWhh,dWhy,dbh,dby,hs[len(inputs)-1]

    def get_samples(self,h,seed_ix,n):
        x  = np.zeros((self.vocab_size,1))
        x[seed_ix] = 1

        ixs = []

        for t in range(n):
            h  = np.tanh(np.dot(self.Wxh,x) + np.dot(self.Whh,h)+self.bh)

            y = np.dot(self.Why, h) + self.by

            p = np.exp(y) /np.sum(np.exp(y))

            ix = np.random.choice(range(self.vocab_size),p = p.ravel())

            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1

            ixs.append(ix)

        return ixs

    def train(self):
        p = 0
        mWxh, mWhh, mWhy = np.zeros_like(self.Wxh),np.zeros_like(self.Whh),np.zeros_like(self.Why)
        mbh,mby = np.zeros_like(self.bh),np.zeros_like(self.by)
        sloss = -np.log(1.0/self.vocab_size)*self.seq_length

        for i in range(self.epoch):
            if p + self.seq_length + 1 >= len(self.data) or i ==0:
                hprev = np.zeros((self.hidden_size,1))
            p = 0

            inputs = [ self.char2index[ch] for ch in self.data[p: p+self.seq_length]]
            targets = [ self.char2index[ch] for ch in self.data[p+1: p+self.seq_length+1]]

            if i%100 ==0:
                sample_ix = self.get_samples(hprev,inputs[0],200)
                text = ''.join(self.index2char[ix] for ix in sample_ix)
                print('====测试样本=====\n')
                print('\n-----\n{}------\n'.format(text))

            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.Compute_LossFunction(inputs,targets,hprev)
            sloss = sloss*0.99 + loss*0.001

            if i%100 == 0:
                print("iter:{}loss:{}".format(i,sloss))

            for params ,dparams ,mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                             [dWxh, dWhh, dWhy, dbh, dby],
                                             [mWxh, mWhh, mWhy, mbh, mby]):
                mem +=dparams * dparams
                params += -self.lr *dparams / np.sqrt(mem + 1e-8)

            p+=self.seq_length



