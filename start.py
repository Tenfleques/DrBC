from BetLearn import BetLearn
import os
import time

def main():
    ########## Train
    NUM_MIN=3000
    NUM_MAX=4000
    weighted=False

    btl = BetLearn(NUM_MIN=NUM_MIN, NUM_MAX=NUM_MAX, weighted=weighted)
    save_dir = './models'
    logs_dir='./logs'

    exp = time.strftime("exp-{}-{}-%Y-%m-%d_%H-%M".format(NUM_MIN, NUM_MAX))

    save_dir = os.path.join(save_dir, exp)
    logs_dir = os.path.join(logs_dir, exp)
    
    btl.Train(save_dir, logs_dir)
    # model_path = "models/paper/nrange_iter_4000_5000_1000.ckpt"
    # data_test = "data/graphs/email-Enron-w.txt"
    # label_file = "data/exact_bc/email-Enron.txt"

    # btl.EvaluateRealData(model_path, data_test, label_file)


if __name__=="__main__":
    main()
