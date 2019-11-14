import multiprocessing
import random
import time
from multiprocessing import Value
from ctypes import c_bool
from get_supernet_distribution import get_flop_param_score


if __name__ == '__main__':
    def trainer(pool, pool_lock, shared_finished_flag):
        iter_left = 10
        while True:
            if iter_left < 1:
                shared_finished_flag.value = True
                break
            time.sleep(2)
            if len(pool) > 1:
                with pool_lock:
                    cand = pool.pop()
                    print('[Trainer]' + '-' * 40)
                    print("Time: {}".format(time.time()))
                    print("Block choice: {}".format(cand['block']))
                    print("Channel choice: {}".format(cand['channel']))
                    print("Flop: {}M, param: {}M".format(cand['flops'], cand['model_size']))
                    iter_left -= 1
            else:
                time.sleep(1)

    def pool_maintainer(pool, pool_lock, shared_finished_flag):
        while True:
            if shared_finished_flag.value:
                break
            if len(pool) < 5:
                candidate = dict()
                candidate['block'] = [random.choice(range(4)) for _ in range(20)]
                candidate['channel'] = [random.choice(range(10)) for _ in range(20)]
                flops, model_size, flop_score, model_size_score = \
                    get_flop_param_score(candidate['block'], candidate['channel'],
                                         use_se=True, last_conv_after_pooling=True)
                candidate['flops'] = flops
                candidate['model_size'] = model_size
                candidate['flop_score'] = flop_score
                candidate['model_size_score'] = model_size_score
                if flop_score > 1:
                    continue
                with pool_lock:
                    pool.append(candidate)
                    print("[Maintainer] Add one good candidate. currently pool size: {}".format(len(pool)))

    # definition
    manager = multiprocessing.Manager()
    cand_pool = manager.list()
    p_lock = manager.Lock()
    finished = Value(c_bool, False)

    process1 = multiprocessing.Process(target=trainer, args=[cand_pool, p_lock, finished])
    process2 = multiprocessing.Process(target=pool_maintainer, args=[cand_pool, p_lock, finished])

    # start work
    for p in [process1, process2]:
        p.start()

    for p in [process1, process2]:
        p.join()

    print(cand_pool)
