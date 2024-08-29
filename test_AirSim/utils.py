# utils.py

import logging

def setup_logging():
    logging.basicConfig(filename='logs/log.txt', level=logging.INFO, format='%(asctime)s %(message)s')
