import argparse

class Hparams():
    """
    hyperparameter passing
    """
    parser = argparse.ArgumentParser()
    
    # dataset
    parser.add_argument("--train", 
                        default=None, 
                        help="training set")
    parser.add_argument("--test", 
                        default=None, 
                        help="test set")
    
    # training scheme
    parser.add_argument("--batch_size",
                        default=8,
                        type=int)
    parser.add_argument("--lr",
                        default=0.1**3,
                        help="learning rate")
    parser.add_argument("--epochs",
                        default=3,
                        help="number of epochs")
    parser.add_argument("--train_dir",
                        default="./train_dir",
                        help="training directory for training result")
    
    # test scheme
    parser.add_argument("--ckpt",
                        default="./ckpt",
                        help="checkpoint")
    parser.add_argument("--test_dir",
                        default="./test_dir",
                        help="test directory for test result")