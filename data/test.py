
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--two', nargs=2, type=int, default=[0, 1, 3, 'a'])
print(parser.parse_args())
