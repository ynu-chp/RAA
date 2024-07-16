import numpy as np
import torch
import os

def generate_data(sample_num,n_agents, m_items,   path):
    value=np.random.rand(sample_num, n_agents, m_items)
    bid=np.random.normal(value,0.1*value)
    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
            for k in range(value.shape[2]):
                while bid[i,j,k]<0 or bid[i,j,k]>2*value[i,j,k]:
                    bid[i,j,k]=np.random.normal(value[i,j,k],0.1*value[i,j,k])
    value = value * 2

    path_dir=os.path.join("data",str(n_agents)+'x'+str(m_items))
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    path_dir = os.path.join(path_dir, path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    np.save(os.path.join(path_dir,"value"), value, allow_pickle=True, fix_imports=True)
    np.save(os.path.join(path_dir, "bid"), bid, allow_pickle=True, fix_imports=True)


print("generate data:")

train_dir="train"
test_dir="test"
final_test_dir="final_test"

n=2
m=5
train_sample_num=32768*2
test_sample_num=32768
final_sample_num=32768
generate_data(int(train_sample_num),int(n),int(m),train_dir)
generate_data(int(test_sample_num),int(n),int(m),test_dir)
generate_data(int(final_sample_num),int(n),int(m),final_test_dir)
print("bidder={},poi={},ok!".format(n,m))

# #
# n=5
# m=4
# train_sample_num=32768*2
# test_sample_num=32768
# final_sample_num=32768
# generate_data(int(train_sample_num),int(n),int(m),train_dir)
# generate_data(int(test_sample_num),int(n),int(m),test_dir)
# generate_data(int(final_sample_num),int(n),int(m),final_test_dir)
# print("bidder={},poi={},ok!".format(n,m))
# # #
# n=5
# m=6
# train_sample_num=32768*2
# test_sample_num=32768
# final_sample_num=32768
# generate_data(int(train_sample_num),int(n),int(m),train_dir)
# generate_data(int(test_sample_num),int(n),int(m),test_dir)
# generate_data(int(final_sample_num),int(n),int(m),final_test_dir)
# print("bidder={},poi={},ok!".format(n,m))
#
# n=5
# m=8
# train_sample_num=32768*2
# test_sample_num=32768
# final_sample_num=32768
# generate_data(int(train_sample_num),int(n),int(m),train_dir)
# generate_data(int(test_sample_num),int(n),int(m),test_dir)
# generate_data(int(final_sample_num),int(n),int(m),final_test_dir)
# print("bidder={},poi={},ok!".format(n,m))
#
# n=5
# m=10
# train_sample_num=32768*2
# test_sample_num=32768
# final_sample_num=32768
# generate_data(int(train_sample_num),int(n),int(m),train_dir)
# generate_data(int(test_sample_num),int(n),int(m),test_dir)
# generate_data(int(final_sample_num),int(n),int(m),final_test_dir)
# print("bidder={},poi={},ok!".format(n,m))
#
#
# n=5
# m=12
# train_sample_num=32768*2
# test_sample_num=32768
# final_sample_num=32768
# generate_data(int(train_sample_num),int(n),int(m),train_dir)
# generate_data(int(test_sample_num),int(n),int(m),test_dir)
# generate_data(int(final_sample_num),int(n),int(m),final_test_dir)
# print("bidder={},poi={},ok!".format(n,m))

# 2x5 -opt-utility: 1.6730597078287701
# 4x5 -opt-utility: 3.2272923216086014
# 6x5 -opt-utility: 4.161047897275068
# 8x5 -opt-utility: 4.4443473062156045
# 10x5 -opt-utility: 4.61249579988953
# 12x5 -opt-utility: 4.729176512049889


# 5x2 -opt-utility: 1.6724045742629414
# 5x4 -opt-utility: 3.228029943743401
# 5x6 -opt-utility: 4.1596277832670125
# 5x8 -opt-utility: 4.4468664701867056
# 5x10 -opt-utility: 4.611975704076485
# 5x12 -opt-utility: 4.7305994834887315

# -----------------budget=1----------------
# 5x2 -opt-utility: 1.2153148182442866
# 2x5 -opt-utility: 1.2144918471342567



# 6x2 -opt-utility: 1.733902631275214
# 6x4 -opt-utility: 3.385749528192491
# 6x6 -opt-utility: 4.859659536821552
# 6x8 -opt-utility: 5.282178247474917
# 6x10 -opt-utility: 5.501998288738378
# 6x12 -opt-utility: 5.654155127185277