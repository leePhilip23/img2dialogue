import logging 

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, 
    format='[%(asctime)s] [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)