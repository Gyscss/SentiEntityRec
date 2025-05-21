# import torch
# print(torch.__version__)  # 输出 PyTorch 版本
# print(torch.cuda.is_available())  # 检查是否可以使用 CUDA


# import subprocess
#
# command = [
#     "pip", "install",
#     "torch==1.13.1+cu117",
#     "torchvision==0.14.1+cu117",
#     "torchaudio==0.13.1",
#     "--extra-index-url", "https://download.pytorch.org/whl/cu117"
# ]
#
# subprocess.run(command)


# import nltk
# nltk.download('punkt_tab')


# import numpy as np
# a1 = np.array([1,2,3,4])
# print(a1) # [1 2 3 4]
# print(type(a1)) # <class 'numpy.ndarray'>
# print(np.__version__)

import os
if os.name == 'nt':  # Windows 环境
    print('nt')
    # os.system("type file.txt")
else:  # Linux/Unix 环境
    print('Linux')
    # os.system("cat file.txt")
