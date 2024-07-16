import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Transformer2DNet(nn.Module):
    def __init__(self, d_input, d_output, n_layer, n_head):

        super(Transformer2DNet, self).__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.n_layer = n_layer

        d_in = d_input
        d_hidden = 4 * d_in

        self.row_transformer = nn.ModuleList()
        self.col_transformer = nn.ModuleList()
        self.fc = nn.ModuleList()
        for i in range(n_layer):
            d_out = d_in if i != n_layer - 1 else d_output

            self.row_transformer.append(nn.TransformerEncoderLayer(d_in, n_head, d_hidden, batch_first=True, dropout=0))
            self.col_transformer.append(nn.TransformerEncoderLayer(d_in, n_head, d_hidden, batch_first=True, dropout=0))
            self.fc.append(nn.Sequential(
                nn.Linear(2 * d_in, d_in),
                nn.ReLU(),
                nn.Linear(d_in, d_out)
            ))

    def forward(self, input):
        bs, n_bidder, n_item, d = input.shape
        x = input
        for i in range(self.n_layer):
            row_x = x.view(-1, n_item, d)
            row = self.row_transformer[i](row_x)
            row = row.view(bs, n_bidder, n_item, -1)

            col_x = x.permute(0, 2, 1, 3).reshape(-1, n_bidder, d)
            col = self.col_transformer[i](col_x)
            col = col.view(bs, n_item, n_bidder, -1).permute(0, 2, 1, 3)

            x = torch.cat([row, col], dim=-1)

            x = self.fc[i](x)
        return x


class TransformerMechanism(nn.Module):
    def __init__(self, n_layer, n_head, d_hidden, menu_size):  # (3   8   64   128)
        super(TransformerMechanism, self).__init__()

        self.pre_net = nn.Sequential(
            nn.Linear(2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )

        self.lambdanet = nn.Sequential(
            nn.Linear(menu_size, menu_size),
            nn.ReLU(),
            nn.Linear(menu_size, menu_size)
        )
        d_input = d_hidden
        self.menu_size = menu_size
        self.n_layer, self.n_head = n_layer, n_head
        self.mechanism = Transformer2DNet(d_input, 2 * menu_size + 1, self.n_layer, self.n_head)

    def forward(self, batch_bid, batch_value,  softmax_temp):
        bid, value = batch_bid, batch_value
        bs, n, m = bid.shape[0], bid.shape[1], bid.shape[2]

        x1 = bid.unsqueeze(-1)  # (bs,n,m,1)
        x2 = value.unsqueeze(-1)  # (bs,n,m,1)

        # print(x4.shape)
        # print(x4)
        # (bs,n,1)

        x = torch.cat([x1, x2], dim=-1)
        x = self.pre_net(x)
        mechanism = self.mechanism(x)
        allocation, b, w = \
            mechanism[:, :, :, :self.menu_size], mechanism[:, :, :, self.menu_size:2 * self.menu_size], mechanism[:, :,
                                                                                                        :, -1]


        alloc1= F.softmax(allocation * softmax_temp, dim=1)  # (bs,n_bidder,m_item,menu_size)//保证物品最多卖一次
        alloc2= F.softmax(allocation * softmax_temp, dim=2)
        alloc = torch.min(alloc1,alloc2)
        # alloc=torch.sigmoid(allocation)
        alloc = alloc.permute(0, 3, 1, 2)  # (bs,menu_size,n_bidder,m_item)
        # alloc = alloc[:, :, :-1, :]  # 去除最后最后一个bidder(虚拟投标人)的分配
        # alloc bs, t, n, m

        w = w.mean(-1)
        w = torch.sigmoid(w)  # (bs,n)
        # w = w[:, :-1]  # 去除最后最后一个bidder(虚拟投标人)的权重
        # w bs, n

        b = b.mean(-2)  # (bs,n_bidder,m_item,menu_size)->(bs,n_bidder,menu_size)
        # b = allocation.mean(-2)
        b = b.mean(-2)  # (bs,n_bidder,menu_size)->(bs,menu_size)
        b = self.lambdanet(b)
        # b bs, t

        return alloc, w, b  # (bs,menu_size,n_bidder,m_item)   (bs,n)  (bs,menu_size)

# Mechanism=TransformerMechanism(3,8,64,32)
# print("yes")
#
# value=torch.randn(3,4,5,)
# bid=torch.randn(3,4,5)
# e=torch.randn(3,4,5)
# E=torch.randn(3,4)
# alloc, w, b=Mechanism(bid,value,e,E,5)
#
# # print(alloc)
# print("alloc.shape:",alloc.shape)
#
# # print(w)
# print("w.shape:",w.shape)
#
# # print(b)
# print("b.shape:",b.shape)
