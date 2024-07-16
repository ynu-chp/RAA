import numpy as np
import torch
from typing import Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
from net import TransformerMechanism
# from IPython import embed


class RAMANet(nn.Module):
    def __init__(self, args, oos=False) -> None:
        super().__init__()
        self.n_agents = args.n_agents
        self.m_items = args.m_items
        # self.dx = args.dx
        # self.dy = args.dy
        self.device = args.device
        self.menu_size = args.menu_size
        self.bool_RVCGNet=args.bool_RVCGNet
        # self.continuous = args.continuous_context #parser.add_argument('--continuous_context', type=str2bool, default=False) 判断是否为连续上下文信息
        # self.const_bidder_weights = args.const_bidder_weights#parser.add_argument('--const_bidder_weights', type=str2bool, default=False) #
        self.alloc_softmax_temperature = args.alloc_softmax_temperature
                                    # parser.add_argument('--init_softmax_temperature', type=int, default=500)
                                    # parser.add_argument('--alloc_softmax_temperature', type=int, default=10)
        mask = 1 - torch.eye((self.n_agents)).to(self.device)#(n,n) torch.eye返回一个二维矩阵，对角线位置为1,其余位置为0, 1-torch.eye则相反
        self.mask = torch.zeros(args.n_agents, args.batch_size, args.n_agents).to(self.device)#(n, bs, n)
        for i in range(args.n_agents):
            self.mask[i] = mask[i].repeat(args.batch_size, 1)
        #self.mask #(n, bs, n)

        self.mask = self.mask.reshape(args.n_agents * args.batch_size, args.n_agents)#(n * bs, n)
        """
        self.mask的形状类似于这种 (n=4,bs=3)
               [[0., 1., 1., 1., 1.],
                [0., 1., 1., 1., 1.],
                [0., 1., 1., 1., 1.],
                [1., 0., 1., 1., 1.],
                [1., 0., 1., 1., 1.],
                [1., 0., 1., 1., 1.],
                [1., 1., 0., 1., 1.],
                [1., 1., 0., 1., 1.],
                [1., 1., 0., 1., 1.],
                [1., 1., 1., 0., 1.],
                [1., 1., 1., 0., 1.],
                [1., 1., 1., 0., 1.],
                [1., 1., 1., 1., 0.],
                [1., 1., 1., 1., 0.],
                [1., 1., 1., 1., 0.]]
        """
        # for ablation study 不同的消融设置
        self.mechanism = TransformerMechanism(args.n_layer, args.n_head, args.d_hidden,
                args.menu_size).to(self.device)
        # self.mechanism = nn.DataParallel(self.mechanism)  # device_ids=[0, 1]
        # V, X, Y -> alloc, mu, lambda

    def test_time_forward(self, input_bids: torch.tensor, input_values: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        '''
        input_bids: B, n, m  (bs,n,m)
        X: B, n_agents, dx  (bs,n,dx)
        Y: B, m_items, dy   (bs,m,dy)
        '''
        B, n, m = input_bids.shape
        value_bid=input_values-input_bids #(bs,n,m)
        # if self.continuous:#这里创建虚拟bidder的上下文信息
        #     X = torch.cat((X, torch.ones(B, 1, self.dx).to(self.device)), axis=1)
        # else:
        #     X = torch.cat((X, torch.ones(B, 1).to(self.device).long() * self.n_agents), axis=1)

        allocs, w, b = self.mechanism(input_bids,input_values,self.alloc_softmax_temperature) #parser.add_argument('--alloc_softmax_temperature', type=int, default=10)
        # (bs,ms,n,m)   (bs,n)  (bs,ms)
        if self.bool_RVCGNet == True: #parser.add_argument('--const_bidder_weights', type=str2bool, default=False) #
            w = torch.ones(w.shape).to(self.device)
            b = torch.zeros(b.shape).to(self.device)
                        # (bs,ms,n,m) (bs, 1, n, m)-> (bs,ms,n,m)
        allocs = torch.cat((allocs, torch.zeros(B, 1, n, m).to(self.device)), 1) # B, t, n, m
                        #(bs,ms) (bs,1) ->(bs,ms)
        b = torch.cat((b, torch.zeros((B, 1)).to(self.device)), 1) # B, t #每个batch_size多加了一种未分配方案(分配为0,提升为0)这样做有什么目的?
        assert w.all() > 0

                          #(bs,ms,n,m)  #(bs,1,n,m)
        util_from_items = (allocs * value_bid.unsqueeze(1)).sum(axis=-1) # B, t, n 通过分配结果计算每个用户得到(价值-报价)
                            #(bs,1,n)       (bs, ms, n)
        per_agent_welfare = w.unsqueeze(1) * util_from_items # B, t, n #用户得到的w(价值-报价)
        total_welfare = per_agent_welfare.sum(axis=-1) # B, t #计算w社会福利
        alloc_choice_ind = torch.argmax(total_welfare + b, -1)  #B, ms中仿射社会福利最大的分配的索引

        item_allocation = [allocs[i, alloc_choice_ind[i],...] for i in range(B)] #得到每个ms中, 仿射社会福利最大的分配[bs,n,m]
        item_allocation = torch.stack(item_allocation) # B, n, m #合并list元素,[bs,n,m]->(bs,n,m)得到每个batch_size中,仿射社会福利最大的分配结果

        chosen_alloc_welfare_per_agent = [per_agent_welfare[i, alloc_choice_ind[i], ...] for i in range(B)] #得到每个ms中, 仿射社会福利最大的分配下用户的w价值[bs,n]
        chosen_alloc_welfare_per_agent = torch.stack(chosen_alloc_welfare_per_agent) # B, n #合并list元素,[bs,n]->(bs,n)得到每个batch_size中,仿射社会福利最大的分配下用户的w价值(bs,n)
        
        ####
        removed_alloc_choice_ind_list = []
        ####
        #计算payment
        payments = []
        for i in range(self.n_agents):#计算每个用户的payment
            mask = torch.ones(n).to(self.device)#(n,)
            mask[i] = 0
                                       #(bs,ms,n)         (1,1,n)
            removed_i_welfare = per_agent_welfare * mask.reshape(1, 1, n)#(bs,ms,n) 去掉用户i的用户w(价值-报价)
            total_removed_welfare = removed_i_welfare.sum(-1) # B, t #(bs,ms)去掉用户i的w社会福利
            removed_alloc_choice_ind = torch.argmax(total_removed_welfare + b, -1) # B #(bs,)去掉用户i,ms中仿射社会福利最大的分配的索引
            removed_chosen_welfare = [total_removed_welfare[i, removed_alloc_choice_ind[i]] for i in range(B)]#[bs],去掉用户i,ms中仿射社会福利最大的分配下的 w社会福利
            removed_chosen_welfare = torch.stack(removed_chosen_welfare)# B #[bs]->(bs,),去掉用户i,ms中仿射社会福利最大的分配下的 w社会福利
                
            removed_alloc_b = [b[i, removed_alloc_choice_ind[i]] for i in range(B)]
            removed_alloc_b = torch.stack(removed_alloc_b) #(bs,) 去掉用户i,ms中仿射社会福利最大的分配 对应的boost

            alloc_b = [b[i, alloc_choice_ind[i]] for i in range(B)]
            alloc_b = torch.stack(alloc_b)#bs  ms中仿射社会福利最大的分配 对应的boost

            payments.append(
                (1.0 / w[:,i]) #(bs,)  w.shape->(bs,n)
                * (
                    (chosen_alloc_welfare_per_agent.sum(1) + alloc_b)
                    -( removed_chosen_welfare + removed_alloc_b )#去掉用户i的最优分配
                )
                    # (bs,n,m)*(bs,n,m)
            )
            removed_alloc_choice_ind_list.append(removed_alloc_choice_ind)#去掉用户i,ms中仿射社会福利最大的分配的索引

        payments = torch.stack(payments)# (n,bs)
                # (bs,n,m)*(bs,n,m)
        payments = payments + (input_bids * item_allocation).sum(-1).permute(1,0)
        utility=(input_values*item_allocation).sum(-1).permute(1,0)-payments
        # ep=torch.relu(item_allocation*input_e-input_E.unsqueeze(-1).repeat(1,1,m)).mean(0) ## (bs,n,m)
        return alloc_choice_ind, item_allocation, utility, payments, allocs, w, b, removed_alloc_choice_ind_list, (input_bids * item_allocation).sum(-1).permute(1,0)#,ep
        #ms中仿射社会福利最大分配的索引(bs,), ms中仿射社会福利最大的分配(bs,n,m), 效用(n,B), 支付方案(n,B), 分配(bs,ms,n,m), 偏置(bs,n), boost(bs,ms), #去掉用户i,ms中仿射社会福利最大的分配的索引(B)
    
    def forward(self, input_bids: torch.tensor, input_values: torch.tensor, softmax_temp: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        '''                      parser.add_argument('--init_softmax_temperature', type=int, default=500)
        input_bids: B, n, m 
        X: B, n_agents, dx 
        Y: B, m_items, dy
        '''
        B, n, m = input_bids.shape
        value_bid = input_values - input_bids  # (bs,n,m)
        # if self.continuous:
        #     X = torch.cat((X, torch.ones(B, 1, self.dx).to(self.device)), axis=1)
        # else:
        #     X = torch.cat((X, torch.ones(B, 1).to(self.device).long()* self.n_agents), axis=1)

        allocs, w, b = self.mechanism(input_bids,input_values,self.alloc_softmax_temperature)#parser.add_argument('--alloc_softmax_temperature', type=int, default=10)
        #(bs,ms,n,m)   (bs,n)   (bs,ms)
        if self.bool_RVCGNet == True: #parser.add_argument('--const_bidder_weights', type=str2bool, default=False) #
            w = torch.ones(w.shape).to(self.device)
            b = torch.zeros(b.shape).to(self.device)

                # (bs,ms,n,m) (bs, 1, n, m)
        allocs = torch.cat((allocs, torch.zeros(B, 1, n, m).to(self.device)), 1) # B, t, n, m
        b = torch.cat((b, torch.zeros((B, 1)).to(self.device)), 1) # B, t #每个batch_size多加了一种未分配方案(分配为0,提升为0)这样做有什么目的?

                        # (bs,ms,n,m) * (bs,1,n,m) .sum(-1) -> (bs,ms,n)
        util_from_items = (allocs * value_bid.unsqueeze(1)).sum(axis=-1) # B, t, n   #计算每个用户得到的(价值-报价)
                        #   (bs,1,n) * (bs,ms,n) ->(bs,ms,n)
        per_agent_welfare = w.unsqueeze(1) * util_from_items # B, t, n #(bs,ms,n)计算每个用户w(价值-报价)
        total_welfare = per_agent_welfare.sum(axis=-1) # B, t  #(bs,ms)计算w社会福利

        # 下面近似计算最大仿射社会福利(温度系数)
                                # sotf: (bs,ms) -> (bs,ms)
        alloc_choice = F.softmax((total_welfare + b) * softmax_temp, dim=-1) # B, t
        # (bs,ms)->(bs,ms,1)->(bs,ms,1,1)* (bs,ms,n,m) .sum(1)-> (bs,n,m)
        item_allocation = (torch.unsqueeze(torch.unsqueeze(alloc_choice, -1), -1) * allocs).sum(axis=1)# (bs,n,m) 近似仿射社会福利最大的分配方案
        # (bs,ms)->(bs,ms,1)*(bs,ms,n) .sum(1) ->(bs,n)
        chosen_alloc_welfare_per_agent = (per_agent_welfare * torch.unsqueeze(alloc_choice, -1)).sum(axis=1) # B, n  #(bs,n)近似最大的社会福利的分配下, 每个用户的w(价值-报价)

        n_chosen_alloc_welfare_per_agent= chosen_alloc_welfare_per_agent.repeat(n, 1)# nB, n  #(n*bs, n)
                                                   #  (n*bs, n) * (n * bs, n) ->  (n * bs, n)
        masked_chosen_alloc_welfare_per_agent = n_chosen_alloc_welfare_per_agent * self.mask #  nB, n #将用户1~n个用户的w价值掩码
                                # (bs,ms,n)->(n*bs,ms,n)
        n_per_agent_welfare = per_agent_welfare.repeat(n, 1, 1)# nB, t, n #(n*bs, ms, n) 每个用户w价值扩展n倍
                                #  (n*bs, ms, n)* (n*bs, 1, n)->(n*bs,ms,n)
        removed_i_welfare = n_per_agent_welfare * self.mask.reshape(n*B, 1, n) # nB, t, n #(n*bs, ms, n)将每种分配的1~n个用户w价值分别掩码
                                #  (n*bs,ms,n).sum(-1) -> (n*bs,ms)
        total_removed_welfare  = removed_i_welfare.sum(axis=-1) # nB, t # (n*bs,ms)
                                #                       (n*bs,ms) * (n*bs,ms) -> (n*bs,ms)
        removed_alloc_choice = F.softmax((total_removed_welfare + b.repeat(n, 1)) * softmax_temp, dim=-1)# (n*bs,ms)
            # nB, t
        removed_chosen_welfare_per_agent = (#近似去掉用户i的其余用户w价值
                #     (n*bs,ms,n) * (n*bs,ms,1)-> (n*bs,ms,n)
            removed_i_welfare * removed_alloc_choice.unsqueeze(-1) # nB, t, n
        ).sum(axis=1)# (n*bs,n)
            # nB, n
        payments = torch.zeros(n * B).to(self.device)
                        # (bs,n)->(n,bs)->(n*bs)
        payments = (1 / w.permute(1, 0).reshape(n * B)) * (
            ( n_chosen_alloc_welfare_per_agent.sum(-1)#最优分配中去掉用户i的w社会福利
            +(alloc_choice * b).sum(1).repeat(n))
            -
            (removed_chosen_welfare_per_agent.sum(-1)#(n*bs,)去掉用户i的最优分配的w社会福利
                #  (n*bs,ms) * (n*bs,ms) .sum(-1) -> (n*bs,)
            + (removed_alloc_choice * b.repeat(n, 1)).sum(-1))
        ) # nB                                 (bs,n,m)
        payments = payments.reshape(n, B)+(item_allocation*input_bids).sum(-1).permute(1,0)
        utility = (input_values * item_allocation).sum(-1).permute(1, 0) - payments
        # ep=torch.relu(item_allocation*input_e-input_E.unsqueeze(-1).repeat(1,1,m)).mean(0)# (bs,n,m)
        return alloc_choice, item_allocation, utility, payments, allocs,(item_allocation*input_bids).sum(-1).permute(1,0)#,ep


