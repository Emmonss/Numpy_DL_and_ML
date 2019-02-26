
import make_data

if __name__ == '__main__':
    list,claser = make_data.MakeData()
    vocab = make_data.CreateList(list)
    print(vocab)