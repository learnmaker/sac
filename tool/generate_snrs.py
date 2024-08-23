import numpy as np
import pandas as pd
import os
import sys
import inspect
import random
# set parent directory as sys path
current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def generate_snrs(server_num):
    
    # 如果有对应信噪比文件，则直接跳过
    flag = True
    for i in range(server_num):
        if os.path.exists("./mydata/temp/dynamic_snrs_" + str(i+1) + ".csv") and os.path.isfile("./mydata/temp/dynamic_snrs_" + str(i+1) + ".csv"):
            continue
        else:
            flag = False
            break
    if flag:
        print("存在对应信噪比文件")
        return
    
    sample_len = 120000 * 4
    snrs = [1, 1.5, 0.5, 2]
    # 每个服务器的信噪比变换是相同的
    for i in range(server_num):
        output = []
        random_snrs=snrs.copy()
        random.shuffle(random_snrs)
        for snr in random_snrs:
            output += [snr] * 1200000
        pd.DataFrame(np.array(output)).to_csv(
            "./mydata/temp/dynamic_snrs_" + str(i+1) + ".csv", header=False, index=False)


if __name__ == '__main__':
    sample_len = 1200000 * 4
    # # generate all one list
    # output = [1 for i in range(sample_len)]
    # pd.DataFrame(np.array(output)).to_csv("./data/one_snrs.csv", header=False, index=False)

    # generate dynamic snrs
    snrs = [1, 3, 0.5, 2]
    output = []
    for snr in snrs:
        output += [snr] * 1200000
    pd.DataFrame(np.array(output)).to_csv(
        "./mydata/dynamic_snrs_4.csv", header=False, index=False)
