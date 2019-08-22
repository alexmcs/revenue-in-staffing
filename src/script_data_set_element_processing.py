import sys
sys.path.append('../src')
import script_data_set_processing
reload(script_data_set_processing)
import logging
import time
import os

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s : %(levelname)s : %(message)s')

    param1 = sys.argv[1]

    if len(sys.argv) == 3:
        param2 = sys.argv[2]

    if not os.path.exists('../data/logs/'):
        os.makedirs('../data/logs/')

    log_file = "log_file_" + param1 + '___' + str(time.time()) + ".log"

    fh = logging.FileHandler("../data/logs/"+log_file, 'w')

    fh.setFormatter(formatter)
    logger.addHandler(fh)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    logger.info('log file ' + log_file + ' was created')

    if len(sys.argv) == 3:
        script_data_set_processing.data_set_element_processing(param1, param2)

    else:
        script_data_set_processing.feature_element_processing(param1)

    logger.info('script finished successfully')