import numpy as np

def batch_index_list(buffer_size, batch_size, repalce=True, batch_num=None):
    indices = [i for i in range(buffer_size)]

    ans = []
    if repalce: # 有放回的采样
        
        if not batch_num:
            batch_num = round(buffer_size / batch_size + 0.5) * 2
        # length = length + 1 if buffer_size % batch_size != 0 else length
        # print (f'length is {batch_num}')
        for i in range(batch_num):
            item = np.random.choice(indices, batch_size, replace=False)
            ans.append(item)
    else:# 无放回的采样
        np.random.shuffle(indices)
        print (indices)
        for i in range(0, buffer_size, batch_size):
            item = indices[i: i + batch_size]
            print (f'item {item}')
            ans.append(item)
    # return ans
    print (ans)

# ans = batch_index_list(10, 2)
# print (ans)
batch_index_list(10, 4)
batch_index_list(10, 4, batch_num=10)
batch_index_list(10, 4, repalce=False)
batch_index_list(10, 10, repalce=False)

def get_batchs_indices(buffer_size, batch_size, repalce=True, batch_num=None):
    indices = [i for i in range(buffer_size)]

    ans = []
    if repalce: # 有放回的采样
        if not batch_num:
            batch_num = round(buffer_size / batch_size + 0.5) * 2
        for _ in range(batch_num):
            batch_index = np.random.choice(indices, batch_size, replace=False)
            ans.append(batch_index)
    else:# 无放回的采样
        np.random.shuffle(indices)
        for i in range(0, buffer_size, batch_size):
            batch_index = indices[i: i + batch_size]
            ans.append(batch_index)
    return ans
