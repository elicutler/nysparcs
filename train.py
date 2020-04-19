
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
  parser.add_argument('--torch_model', type=str)
  parser.add_argument('--sklearn_model', type=str)
  parser.add_argument('--local_data_path', type=str)
  parser.add_argument('--socrata_data_key', type=str)
  parser.add_argument(
    '--train_from_scratch', action='store_true', 
    help=(
      'If set, do _not_ attempt to load model parameters from prior run'
      ' to initialize parameters for current run.'
    )
  )
  parser.add_argument('--epochs', type=int, default=1)
  parser.add_argument(
    '--batch_size', type=int, default=-1, help='-1: do not batch data'
  )
  parser.add_argument(
    '--n_workers', type=int, default=-1, help='-1: use all but one core'
  )
  args = parser.parse_args()
  
  assert bool(args.torch_model) + bool(args.sklearn_model) == 1
  assert bool(args.local_data_path) + bool(args.socrata_data_key) == 1
  
  modelType = 'torch' if args.torch_model is not None else 'sklearn'
  modelName = args.torch_model or args.sklearn_model
  
  dataLoc = 'local' if args.local_data_path is not None else 'internet'
  dataID = args.local_data_path or args.socrata_data_key
    
  trainer = TrainerFactory.make(
    modelType, modelName, dataLoc, dataID, args.batch_size, args.n_workers,
    trainFromScratch=args.train_from_scratch
  )
  trainer.train(args.epochs)
  trainer.saveModel()
  
  if args.deploy:
    trainer.deployModel()