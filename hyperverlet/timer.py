import time


def timer(func, desc):
    t1 = time.time()
    res = func()
    t2 = time.time()
    print(f'{desc} took: {t2 - t1}s')

    return res
