import logging
from datetime import datetime as dt
from uuid import uuid4

TIME_FORMAT = "%Y%m%d_%H%M%S"
MESSAGE_FORMAT = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s'
FILE = f"output_{uuid4().hex}_{dt.today().strftime(TIME_FORMAT)}.log"

logging.basicConfig(
    filemode='a',
    filename=FILE,
    format=MESSAGE_FORMAT,
    datefmt=TIME_FORMAT,
    level=logging.INFO
)

logger = logging.getLogger("PRUNE_EXPERIMENTS_RUNNER")