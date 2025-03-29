from argparse import ArgumentParser

def args_init():
    parser = ArgumentParser(description='Arguments for training and evaluation')
    return parser

def trainer_args(parser:ArgumentParser):
    parser.add_argument(
        '--dataset',
        type=str,
        default='clean_labeled_contracts.json',
        help='Path to the dataset.'
    )
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=3,
        help='Number of epochs to train.'
    )
    parser.add_argument(
        '--c_lr',
        type=float,
        default=1e-3,
        help='Learning rate for the optimizer.'
    )
    parser.add_argument(
        '--r_lr',
        type=float,
        default=5e-6,
        help='Learning rate for the optimizer.'
    )