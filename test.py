import multiprocessing
import time

def worker(index):
    print("进入函数")
    """worker function"""
    print(f'Worker {index} started')
    time.sleep(2)  # 模拟耗时操作
    print(f'Worker {index} finished')

if __name__ == '__main__':
    # 创建一个进程列表
    processes = []

    # 创建并启动两个进程
    for i in range(10):
        p = multiprocessing.Process(target=worker, args=(i,))
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()

    print('All workers finished.')