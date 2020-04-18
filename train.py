
import logging

from argparse import ArgumentParser
from trainer import TrainerFactory

logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO
)
logger = logging.getLogger(__name__)


if __name__ == '__main__':

  parser = ArgumentParser()
  parser.add_argument(
    '--model_type', type=str, default='torch', choices=['torch', 'sklearn']
  )
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--batch_size', type=int, default=100)
  parser.add_argument('--init_params_from_scratch', action='store_true')
  args = parser.parse_args()
  
  trainer = TrainerFactory.make(args.model_type)
  trainer.train(args.epochs, args.batch_size)
  trainer.saveModel()
  
  if args.deploy:
    trainer.deployModel()