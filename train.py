
from argparse import ArgumentParser
from trainer import TrainerFactory

if __name__ == '__main__':

  parser = ArgumentParser()
  parser.add_argument(
    '--model_type', type=str, default='torch', choices=['torch', 'sklearn']
  )
  args = parser.parse_args()
  
  trainer = Trainer(args)
  trainer.train()
  trainer.saveModel()
  
  if args.deploy:
    trainer.deployModel()