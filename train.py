
from argparse import ArgumentParser
from trainer import Trainer

if __name__ == '__main__':

  parser = ArgumentParser()
  args = parser.parse_args()
  
  trainer = Trainer(args)
  trainer.train()
  trainer.saveModel()
  
  if args.deploy:
    trainer.deployModel()