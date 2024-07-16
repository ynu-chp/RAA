
import numpy as np
import torch
from typing import Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import logging

# def generate_all_deterministic_alloc(n_agents, m_items,unit_demand = False) -> torch.tensor: # n buyers, m items -> alloc (n+1, m)
#     alloc_num = (n_agents+1) ** (m_items)
#     def gen(t, i, j):
#         x = (n_agents+1) ** (m_items - 1 - j)
#         return np.where((t // x) % (n_agents+1) == i, 1.0, 0.0)
#     alloc = np.fromfunction(gen, (alloc_num-1, n_agents, m_items))
#     return torch.tensor(alloc).to(torch.float32)
#
# def filter_tensor(t: torch.Tensor) -> torch.Tensor:
#     # 计算每个矩阵的行和列的和
#     row_sum = t.sum(dim=2)
#     col_sum = t.sum(dim=1)
#
#     # 确定要保留的矩阵索引
#     keep_indices = ((row_sum <= 1).all(dim=1) & (col_sum <= 1).all(dim=1))
#
#     # 使用保留的索引从原始张量中选择子集
#     filtered_tensor = t[keep_indices]
#
#     return filtered_tensor
def generate_all_deterministic_alloc(n, m):
    x = np.zeros((n, m))
    flag = np.zeros(m)
    result = []

    def dfs(u):
        if u >= n:
            # print(x)
            # print()
            result.append(x.tolist())
            return
        dfs(u + 1)
        for i in range(m):
            if flag[i] == 0:
                flag[i] = True
                x[u][i] = 1
                dfs(u + 1)
                x[u][i] = 0
                flag[i] = False


    dfs(0)
    if not os.path.exists("alloc"):
        os.makedirs("alloc")
    # if not os.path.exists(os.path.join("alloc",str(n)+"x"+str(m))):
    #     os.makedirs(os.path.join("alloc",str(n)+"x"+str(m)))
    np.save(os.path.join("alloc",str(n)+"x"+str(m)), np.array(torch.tensor(result).to(torch.float32)), allow_pickle=True, fix_imports=True)
    return torch.tensor(result).to(torch.float32)

def load_data(dir):
    data = [np.load(os.path.join(dir, 'bid.npy')).astype(np.float32),
            np.load(os.path.join(dir, 'value.npy')).astype(np.float32)]
    return torch.tensor(np.array(data)).to(torch.float32)

def test_time_forward( input_bids: torch.tensor, input_values: torch.tensor):
    """
    input_bids: B, n, m  (bs,n,m)
    X: B, n_agents, dx  (bs,n,dx)
    Y: B, m_items, dy   (bs,m,dy)
    """
    B, n, m = input_bids.shape
    value_bid =input_values -input_bids  # (bs,n,m)
    # if self.continuous:#这里创建虚拟bidder的上下文信息
    #     X = torch.cat((X, torch.ones(B, 1, self.dx).to(self.device)), axis=1)
    # else:
    #     X = torch.cat((X, torch.ones(B, 1).to(self.device).long() * self.n_agents), axis=1)

    # allocs, w, b = self.mechanism(input_bids ,input_values, self.alloc_softmax_temperature)  # parser.add_argument('--alloc_softmax_temperature', type=int, default=10)
    # (bs,ms,n,m)   (bs,n)  (bs,ms)
    # if self.bool_RVCGNet == True:  # parser.add_argument('--const_bidder_weights', type=str2bool, default=False) #
    # print("1111111111111")
    if os.path.exists(os.path.join("alloc",str(n)+"x"+str(m)+".npy")):
        allocs=torch.tensor(np.load(os.path.join("alloc",str(n)+"x"+str(m)+".npy")).astype(np.float32))
    else :
        allocs=generate_all_deterministic_alloc(n,m)
        # print("11111111111")

    allocs=allocs.unsqueeze(0).repeat(B,1,1,1).to(DEVICE)

    # (bs,ms,n,m)  #(bs,1,n,m)
    util_from_items = (allocs * value_bid.unsqueeze(1)).sum(axis=-1) # B, t, n 通过分配结果计算每个用户得到(价值-报价)
    # (bs,1,n)       (bs, ms, n)
    per_agent_welfare =  util_from_items # B, t, n #用户得到的w(价值-报价)
    total_welfare = per_agent_welfare.sum(axis=-1) # B, t #计算w社会福利
    alloc_choice_ind = torch.argmax(total_welfare , -1)  # B, ms中仿射社会福利最大的分配的索引

    item_allocation = [allocs[i, alloc_choice_ind[i] ,...] for i in range(B)]  # 得到每个ms中, 仿射社会福利最大的分配[bs,n,m]
    item_allocation = torch.stack(item_allocation) # B, n, m #合并list元素,[bs,n,m]->(bs,n,m)得到每个batch_size中,仿射社会福利最大的分配结果

    chosen_alloc_welfare_per_agent = [per_agent_welfare[i, alloc_choice_ind[i], ...] for i in range(B)]  # 得到每个ms中, 仿射社会福利最大的分配下用户的w价值[bs,n]
    chosen_alloc_welfare_per_agent = torch.stack \
        (chosen_alloc_welfare_per_agent) # B, n #合并list元素,[bs,n]->(bs,n)得到每个batch_size中,仿射社会福利最大的分配下用户的w价值(bs,n)

    ####
    removed_alloc_choice_ind_list = []
    ####
    # 计算payment
    payments = []
    for i in range(n)  :  # 计算每个用户的payment
        mask = torch.ones(n) # (n,)
        mask[i] = 0
        # (bs,ms,n)         (1,1,n)
        removed_i_welfare = per_agent_welfare * mask.reshape(1, 1, n  ).to(DEVICE)  # (bs,ms,n) 去掉用户i的用户w(价值-报价)
        total_removed_welfare = removed_i_welfare.sum(-1) # B, t #(bs,ms)去掉用户i的w社会福利
        removed_alloc_choice_ind = torch.argmax(total_removed_welfare , -1) # B #(bs,)去掉用户i,ms中仿射社会福利最大的分配的索引
        removed_chosen_welfare = [total_removed_welfare[i, removed_alloc_choice_ind[i]] for i in range(B)  ]  # [bs],去掉用户i,ms中仿射社会福利最大的分配下的 w社会福利
        removed_chosen_welfare = torch.stack(removed_chosen_welfare  )# B #[bs]->(bs,),去掉用户i,ms中仿射社会福利最大的分配下的 w社会福利

        # removed_alloc_b = [b[i, removed_alloc_choice_ind[i]] for i in range(B)]
        # removed_alloc_b = torch.stack(removed_alloc_b)  # (bs,) 去掉用户i,ms中仿射社会福利最大的分配 对应的boost
        #
        # alloc_b = [b[i, alloc_choice_ind[i]] for i in range(B)]
        # alloc_b = torch.stack(alloc_b  )  # bs  ms中仿射社会福利最大的分配 对应的boost

        payments.append(
             (
                    (chosen_alloc_welfare_per_agent.sum(1)) -( removed_chosen_welfare )
            # 去掉用户i的最优分配
            )
            # (bs,n,m)*(bs,n,m)
        )
        removed_alloc_choice_ind_list.append(removed_alloc_choice_ind)# 去掉用户i,ms中仿射社会福利最大的  分配的索引

    payments = torch.stack(payments)# (n,bs)
    # (bs,n  ,m)*(bs,n,m)
    payments = payments + (input_bids * item_allocation).sum(-1).permute(1,0)
    utility=(input_values*item_allocation).sum (-1).permute(1,0)-payments
    return utility, payments, (input_bids * item_allocation).sum(-1).permute(1,0)


n=int(5)
m=int(12)
bs=int(256)
test_dir="train"
path_dir=os.path.join("data",str(n)+'x'+str(m))
path_dir = os.path.join(path_dir, test_dir)
test_data = load_data(path_dir)

DEVICE="cuda:0"

test_bids = test_data[0]
test_values = test_data[1]
test_utility = torch.zeros(1)
test_payment = torch.zeros(1)
test_cost = torch.zeros(1)
for num in tqdm(range(int(test_values.shape[0] / bs))):
    utility, payment,  cost = test_time_forward(
        test_bids[num * bs:(num + 1) * bs].to(DEVICE) ,
        test_values[num * bs:(num + 1) * bs].to(DEVICE))
    test_utility += utility.sum().cpu().data
    test_payment += payment.sum().cpu().data
    test_cost += cost.sum().cpu().data
test_utility /= test_values.shape[0]
test_payment /= test_values.shape[0]
test_cost /= test_values.shape[0]
print(f"RVCG:n={n},m={m}, test_vsp-utility: {test_utility}," f"test_users-utility: {test_payment-test_cost}")