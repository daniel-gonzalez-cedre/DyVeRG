import time


def timer(func, logger):
    def wrapper(*args, **kwargs):
        start = time.process_time()
        result = func(*args, **kwargs)
        end = time.process_time()
        logger.info('total time elapsed: {time_elapsed}', time_elapsed=(end - start))
        return result
    return wrapper


def fit_timer(func, logger):
    def wrapper(*args, **kwargs):
        start = time.process_time()
        result = func(*args, **kwargs)
        end = time.process_time()
        logger.info('fit time elapsed: {time_elapsed}', time_elapsed=(end - start))
        return result
    return wrapper


def gen_timer(func, logger):
    def wrapper(*args, **kwargs):
        start = time.process_time()
        result = func(*args, **kwargs)
        end = time.process_time()
        logger.info('gen time elapsed: {time_elapsed}', time_elapsed=(end - start))
        return result
    return wrapper
