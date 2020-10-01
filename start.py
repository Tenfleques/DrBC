from BetLearn import BetLearn
import os
import time

def main():
    ########## Train
    btl = BetLearn(NUM_MIN=30, NUM_MAX=40)
    save_dir = './models'
    logs_dir='./logs'

    exp = time.strftime("exp-%Y-%m-%d_%H-%M")

    save_dir = os.path.join(save_dir, exp)
    logs_dir = os.path.join(logs_dir, exp)
    
    btl.Train(save_dir, logs_dir)
    # model_path = "models/paper/nrange_iter_4000_5000_1000.ckpt"
    # data_test = "/Users/tendai/dev/betweenes-centrality/data/graphs/email-Enron-w.txt"
    # label_file = "/Users/tendai/dev/betweenes-centrality/data/exact_bc/email-Enron.txt"

    # btl.EvaluateRealData(model_path, data_test, label_file)


if __name__=="__main__":
    main()
