import os
import datetime
import argparse
import random
import numpy as np
import torch
import logging
import logging.config


class OptInit():
    """
    Option initialization class.
    This class is responsible for:
    - Parsing command-line arguments
    - Setting up the computing device (CPU/GPU)
    - Initializing random seeds for reproducibility
    """

    def __init__(self):
        """
        Initialize argument parser and parse command-line arguments.
        """
        parser = argparse.ArgumentParser(
            description='PyTorch implementation of PKHG-GCN'
        )

        # Training or evaluation mode
        parser.add_argument(
            '--train', default=0, type=int,
            help='training mode (1) or evaluation mode (0, default)'
        )

        # Whether to force using CPU
        parser.add_argument(
            '--use_cpu', action='store_true',
            help='use CPU instead of GPU'
        )

        # Model architecture parameters
        parser.add_argument(
            '--hgc', type=int, default=16,
            help='number of hidden units in the graph convolution layer'
        )
        parser.add_argument(
            '--lg', type=int, default=5,
            help='number of graph convolution layers'
        )
        parser.add_argument(
            '--lg1', type=int, default=5,
            help='number of graph convolution layers (secondary branch)'
        )

        # Optimization parameters
        parser.add_argument(
            '--lr', default=0.1, type=float,
            help='initial learning rate'
        )
        parser.add_argument(
            '--wd', default=5e-5, type=float,
            help='weight decay'
        )
        parser.add_argument(
            '--num_iter', default=1000, type=int,
            help='number of training epochs'
        )

        # Regularization parameters
        parser.add_argument(
            '--edropout', type=float, default=0.4,
            help='edge dropout rate'
        )
        parser.add_argument(
            '--dropout', default=0.1, type=float,
            help='dropout ratio'
        )

        # Classification setting
        parser.add_argument(
            '--num_classes', type=int, default=2,
            help='number of output classes'
        )

        # Checkpoint saving path
        parser.add_argument(
            '--ckpt_path', type=str,
            default='./save_models/PKHG-GCN',
            help='path to save trained model checkpoints'
        )

        # Parse arguments
        args = parser.parse_args()

        # Add timestamp for experiment identification
        args.time = datetime.datetime.now().strftime("%y%m%d")

        # Set computation device
        if args.use_cpu:
            args.device = torch.device('cpu')
        else:
            args.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu'
            )
            print("Using GPU in torch")

        self.args = args

    def print_args(self):
        """
        Print all configuration parameters to the console.
        """
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}: {}".format(arg, content))
        print("==========     CONFIG END    =============")
        print("\n")

        # Indicate current phase
        phase = 'train' if self.args.train == 1 else 'eval'
        print('===> Phase is {}.'.format(phase))

    def initialize(self):
        """
        Initialize experiment settings:
        - Set random seed
        - Print configuration parameters
        """
        self.set_seed(123)
        # self.logging_init()  # logging can be initialized here if needed
        self.print_args()
        return self.args

    def set_seed(self, seed=0):
        """
        Set random seeds for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
