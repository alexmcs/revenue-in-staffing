import sys
sys.path.append('../src')
import modeling_and_predicting
reload(modeling_and_predicting)
import logging
import time
import os

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s : %(levelname)s : %(message)s')

    param1 = sys.argv[1]

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

    try:

        if len(sys.argv) == 3:
            param2 = sys.argv[2]
            exec ('modeling_and_predicting.' + param1 + '(window="' + param2 + '")')

        else:
            exec ('modeling_and_predicting.' + param1 + '()')

    except Exception as e:
        logger.error("SOMETHING WENT WRONG!!!", exc_info=True)
        raise

    logger.info('script finished successfully')