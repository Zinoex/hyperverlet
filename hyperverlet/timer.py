import time


def timer(func, desc, return_time=False):
    t1 = time.time()
    res = func()
    t2 = time.time()

    total_time = t2 - t1

    print(f'{desc} took: {total_time}s')

    if return_time:
        return res, total_time

    return res
