from BetLearn import BetLearn


def main():
    ########## Train
    btl = BetLearn()
    # btl.Train()
    model_path = "models/paper/nrange_iter_4000_5000_1000.ckpt"
    data_test = "/Users/tendai/dev/betweenes-centrality/data/graphs/email-Enron-w.txt"
    label_file = "/Users/tendai/dev/betweenes-centrality/data/exact_bc/email-Enron.txt"

    btl.EvaluateRealData(model_path, data_test, label_file)


if __name__=="__main__":
    main()
