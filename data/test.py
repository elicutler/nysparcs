
import argparse

p = argparse.ArgumentParser()
p.add_argument('--a', default='ay')
p.add_argument('--b', default='bee')

x = p.parse_args(['--a', 'new ay'])
print(x)
