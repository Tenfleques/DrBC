from BetLearn import BetLearn
import os
import time

def main():
    ########## Train
    NUM_MIN=300000
    NUM_MAX=400000
    weighted=False

    btl = BetLearn(NUM_MIN=NUM_MIN, NUM_MAX=NUM_MAX, weighted=weighted)
    save_dir = './data/exp-30000-40000-2020-10-02_18-41'
    # save_dir = './models/exp-3000-4000-2020-10-01_23-08'
    logs_dir='./logs'

    exp = time.strftime("exp-{}-{}-%Y-%m-%d_%H-%M".format(NUM_MIN, NUM_MAX))

    # save_dir = os.path.join(save_dir, exp)
    logs_dir = os.path.join(logs_dir, exp)
    
    btl.Train(save_dir, logs_dir)
    # model_path = "models/exp-300-400-2020-10-02_12-48/checkpoints/nrange_iter_300_400_9500.ckpt"
    # data_test = "../data/graphs/email-Enron.txt"
    # label_file = "../data/exact_bc/email-Enron.txt"

    # top001, top005, top01, kendal, run_time = btl.EvaluateRealData(model_path, data_test, label_file)

    # print("top001, top005, top01, kendal, run_time \n {}, {}, {}, {} {}".format(top001, top005, top01, kendal, run_time))


if __name__=="__main__":
    main()
