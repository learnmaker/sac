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
            print("生成信噪比文件中...")
            break
    if flag:
        print("存在对应信噪比文件")
        return
    
    snrs = [1, 1.25, 1.5, 1.75, 2]
    
    # 每个服务器的信噪比变换是相同的
    for i in range(server_num):
        output = []
        random_snrs=snrs.copy()
        # 打乱random_snrs
        random.shuffle(random_snrs)
        for snr in random_snrs:
            output += [snr] * 100000
        pd.DataFrame(np.array(output)).to_csv(
            "./mydata/temp/dynamic_snrs_" + str(i+1) + ".csv", header=False, index=False)
