import os
import argparse
import math
import csv

import graphformer_trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        choices=['baseline','gcn','gat','chebnet','appnp','sgcn'],
                        default='baseline',
                        help='Which model to run')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Running model: {args.model} for {args.epochs} epochs")
    
    # Call the main function from graphformer_trainer.py
    graphformer_trainer.main(args.epochs)
    
    
if __name__=='__main__':
    main()