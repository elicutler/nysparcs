
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--a', type=int)
parser.add_argument('--b', type=int)

args = parser.parse_args()

print(args)
