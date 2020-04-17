
from train_arg_parser import TrainArgParser
from trainer import Trainer

if __name__ == '__main__':
  args = TrainArgParser().parse()
  trainer = Trainer(args)
  trainer.fit()
  trainer.saveModel()
  
  if args.deploy:
    trainer.deployModel()