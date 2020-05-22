
import typing as T
import logging

logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
  
  logger.info('Begin deployment...')
  
  parser = DeployArgParser()
  params = parser.parseArgs()
  
  deployer = DeployerFactory.make(params)
  deployer.deploy()
  
  logger.info('Deployment complete')