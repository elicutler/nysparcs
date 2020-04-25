
import logging

from train_arg_parser import TrainArgParser
from safe_dict import SafeDict
from trainer import TrainerFactory

logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO
)
logger = logging.getLogger(__name__)


if __name__ == '__main__':

  parser = TrainArgParser()
  params = SafeDict.fromNamespace(parser.parseArgs(['--run_id', 'fake']))
  
  trainer = TrainerFactory.make(params)
  trainer.train()
  trainer.saveModel()
  
  if args.deploy:
    trainer.deployModel()
