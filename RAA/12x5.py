import argparse
import torch
from auction import RAMANet
from tqdm import tqdm
import logging
import os
import numpy as np
from logger import get_logger


# logging.basicConfig(
# level=logging.INFO,
# format="%(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s",
# datefmt="%Y-%m-%d,%H:%M:%S",
# )

def str2bool(v):
    return v.lower() in ('true', '1')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/12x5')
    parser.add_argument('--training_set', type=str, default='train')
    parser.add_argument('--test_set', type=str, default='test')
    parser.add_argument('--final_test_set', type=str, default='final_test')

    parser.add_argument('--n_agents', type=int, default=12)
    parser.add_argument('--m_items', type=int, default=5)
    parser.add_argument('--menu_size', type=int, default=128)
    parser.add_argument('--bool_RVCGNet', type=str2bool, default=False)
    #gama
    parser.add_argument('--gama', type=float, default=1)
    parser.add_argument('--rho_gama', type=float, default=1)
    parser.add_argument('--rho_gama_update_freq', type=int, default=5000)
    parser.add_argument('--delta_rho_gama', type=float, default=1)

    # ep
    # parser.add_argument('--lamb_ep', type=float, default=0.5)
    # parser.add_argument('--lamb_ep_update_freq', type=int, default=20)
    # parser.add_argument('--rho_ep', type=float, default=0.5)
    # parser.add_argument('--rho_ep_update_freq', type=int, default=100)
    # parser.add_argument('--delta_rho_ep', type=float, default=0.5)

    parser.add_argument('--n_layer', type=int, default=5)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--d_hidden', type=int, default=64)
    parser.add_argument('--init_softmax_temperature', type=int, default=500)
    parser.add_argument('--alloc_softmax_temperature', type=int, default=1)

    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--train_steps', type=int, default=4000)
    parser.add_argument('--train_sample_num', type=int, default=32768 * 2)
    parser.add_argument('--eval_freq', type=int, default=50)  # 500
    parser.add_argument('--eval_sample_num', type=int, default=32768)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--device', type=str, default='cuda:2')

    parser.add_argument('--lr', type=float, default=3e-4)#3e-4
    parser.add_argument('--decay_round_one', type=int, default=2000)  #
    parser.add_argument('--one_lr', type=float, default=5e-5)  #
    parser.add_argument('--decay_round_two', type=int, default=3000)  #
    parser.add_argument('--two_lr', type=float, default=1e-5)  #
    parser.add_argument('--bool_test', type=str2bool, default=False)
    parser.add_argument('--name', type=str, default='./results')
    parser.add_argument('--final_test_batch_size', type=int, default=2048)

    return parser.parse_args()


def load_data(dir):
    data = [np.load(os.path.join(dir, 'bid.npy')).astype(np.float32),
            np.load(os.path.join(dir, 'value.npy')).astype(np.float32)]

    return tuple(data)


if __name__ == "__main__":
    args = parse_args()

    file_path = f"{args.n_agents}_{args.m_items}_{args.menu_size}"
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    log_path = f"{file_path}/record.log"
    logger = get_logger(log_path)
    logger.info(args)

    torch.manual_seed(args.seed)
    DEVICE = args.device

    train_dir = os.path.join(args.data_dir, args.training_set)  #get train data
    train_data = load_data(train_dir)

    test_dir = os.path.join(args.data_dir, args.test_set)  # get test data
    test_data = load_data(test_dir)

    final_test_dir = os.path.join(args.data_dir, args.final_test_set)  # get final test data
    final_test_data = load_data(final_test_dir)

    rho_gama=args.rho_gama
    # rho_ep = args.rho_ep
    # lamb_ep = args.lamb_ep * torch.ones(1).to(DEVICE)

    model = RAMANet(args).to(DEVICE)
    # if args.bool_test:
    # state_dict = torch.load('model/2x5')
    # model.mechanism.load_state_dict(state_dict)

    cur_softmax_temperature = args.init_softmax_temperature
    warm_up_init = 1e-8
    warm_up_end = args.lr
    warm_up_anneal_increase = (warm_up_end - warm_up_init) / 100
    optimizer = torch.optim.Adam(model.mechanism.parameters(), lr=warm_up_init)

    bs = args.batch_size
    num_per_train = int(
        args.train_sample_num / bs)  # parser.add_argument('--train_sample_num', type=int, default = 32768)
    for i in tqdm(range(args.train_steps)):  # parser.add_argument('--train_steps', type=int, default=2000)
        if i == args.train_steps - 1 or (i >= 1000 and (
                i % args.eval_freq == 0)):  # eval parser.add_argument('--eval_freq', type=int, default=500)
            if i == args.train_steps - 1:  # 保存模型
                if not os.path.exists("model"):
                    os.makedirs("model")
                # if not os.path.exists(os.path.join("model", "budget=" + str(args.budget))):
                #     os.makedirs(os.path.join("model", "budget=" + str(args.budget)))

                model_path =os.path.join("model",
                                                       str(args.n_agents) + "x"+str(args.m_items))
                torch.save(model.mechanism.state_dict(), model_path)
            with torch.no_grad():
                # test_values, test_X, test_Y = my_generate_sample(args.eval_sample_num, # parser.add_argument('--eval_sample_num', type=int, default = 32768)
                #                                                  args.n_agents, args.m_items,
                #                                                  args.dx, args.dy, DEVICE)
                test_bids = test_data[0]
                test_values = test_data[1]
                test_utility = torch.zeros(1).to(DEVICE)
                test_payment = torch.zeros(1).to(DEVICE)
                test_cost = torch.zeros(1).to(DEVICE)
                for num in range(int(test_values.shape[0] / bs)):
                    choice_id, _, utility, payment, allocs, _, _, _, cost = model.test_time_forward(
                        torch.tensor(test_bids[num * bs:(num + 1) * bs]).to(DEVICE),
                        torch.tensor(test_values[num * bs:(num + 1) * bs]).to(DEVICE))
                    test_utility += utility.sum()
                    test_payment += payment.sum()
                    test_cost += cost.sum()
                test_utility /= test_values.shape[0]
                test_payment /= test_values.shape[0]
                test_cost /= test_values.shape[0]
                logger.info(
                    f"step {i}: test_sp_utility: {test_utility}," f"test_user_utility: {test_payment-test_cost}," f"test_payment: {test_payment}," f"test_cost: {test_cost}")

        # train_values, train_X, train_Y = my_generate_sample(args.train_sample_num,
        #                                                     args.n_agents, args.m_items,
        #                                                     args.dx, args.dy, DEVICE)
        train_bids = train_data[0]
        train_values = train_data[1]
        reportloss = 0
        train_utility = 0
        train_payment = 0
        train_cost = 0
        train_op=0
        for num in range(num_per_train):  # train num_per_train = int(args.train_sample_num / bs)
            optimizer.zero_grad()  # parser.add_argument('--batch_size', type=int, default = 2048)
            # parser.add_argument('--train_sample_num', type=int, default = 32768)
            _, _, utility, payment, allocs, cost = model(torch.tensor(train_bids[num * bs:(num + 1) * bs]).to(DEVICE),
                                                         torch.tensor(train_values[num * bs:(num + 1) * bs]).to(DEVICE),
                                                         cur_softmax_temperature)

            # print(ep.shape)
            # if args.gama>=1:
            #     op=(1/args.gama)*(torch.abs(utility.mean(0)-args.gama*(payment.mean(0)-cost.mean(0))).mean())
            # else :
            #     op=args.gama*(torch.abs((1.0/args.gama)*utility.mean(0)-(payment.mean(0)-cost.mean(0))).mean())
            # if args.gama>=1:
            #     op=(1/args.gama)*((utility.sum(0)-args.gama*(payment.sum(0)-cost.sum(0))).mean())
            # else :
            #     op=args.gama*(torch.abs((1.0/args.gama)*utility.sum(0)-(payment.sum(0)-cost.sum(0))).mean())
            # lagrangianLoss_ep = torch.sum(op * lamb_ep)
            # # print(ep)
            # lagLoss_ep = (rho_ep / 2) * torch.sum(torch.pow(op, 2))
            # print(args.gama)
            # if i<=500 :
            #     loss = - 1/args.n_agents*utility.sum(0).mean()
            # else:
            # loss = - (utility.sum(0).mean() + (payment.sum(0).mean() - cost.sum(0).mean())) +op
            # loss = - (utility.sum(0).mean()+(payment.sum(0).mean()-cost.sum(0).mean()))+0.5*torch.abs(args.gama*(utility.sum(0)+payment.sum(0)-cost.sum(0))-utility.sum(0)).mean()+0.5*torch.abs((1-args.gama)*(utility.sum(0)+payment.sum(0)-cost.sum(0))-(payment.sum(0)-cost.sum(0))).mean()#rho_gama*op#lagrangianLoss_ep+lagLoss_ep
            loss = (- (utility.sum(0).mean() + (payment.sum(0).mean() - cost.sum(0).mean())) \
                   + 0.5 * torch.abs(args.gama * (utility.sum(0) + payment.sum(0) - cost.sum(0)) - utility.sum(0)).mean() \
                   + 0.5 * torch.abs((1 - args.gama) * (utility.sum(0) + payment.sum(0) - cost.sum(0)) - (payment.sum(0) - cost.sum(0))).mean())
            reportloss += loss.data
            train_utility += utility.sum(0).mean().data
            train_payment += payment.sum(0).mean().data
            train_cost += cost.sum(0).mean().data

            # train_op+=op.data
            loss.backward()
            optimizer.step()

        if i % 1 == 0:
            logger.info(f"step {i}: loss: {reportloss / num_per_train},"
                        f"train_sp_utility: {train_utility / num_per_train},"
                        f"train_user_utility: {(train_payment -train_cost)/ num_per_train},"
                        f"train_payment: {train_payment / num_per_train}," 
                        f"train_cost: {train_cost / num_per_train}")
            # print(train_op/num_per_train)

        if i <= 100:  # warm up
            for p in optimizer.param_groups:
                p['lr'] += warm_up_anneal_increase  # 预热

        if i == args.decay_round_one:  # parser.add_argument('--decay_round_one', type=int, default = 3000) #
            for p in optimizer.param_groups:
                p['lr'] = args.one_lr  # parser.add_argument('--one_lr', type=float, default = 5e-5) #

        if i == args.decay_round_two:  # parser.add_argument('--decay_round_two', type=int, default = 6000) #
            for p in optimizer.param_groups:
                p['lr'] = args.two_lr  # parser.add_argument('--two_lr', type=float, default = 1e-5) #

        # if  i % args.lamb_ep_update_freq ==0:
        #     lamb_ep += rho_ep* train_op
        #
        if i % args.rho_gama_update_freq == 0:
            rho_gama += args.delta_rho_gama

    # test
    logger.info("------------Final test------------")
    with torch.no_grad():
        # test_values, test_X, test_Y = my_generate_sample(args.eval_sample_num, # parser.add_argument('--eval_sample_num', type=int, default = 32768)
        #                                                  args.n_agents, args.m_items,
        #                                                  args.dx, args.dy, DEVICE)
        final_test_bids = final_test_data[0]
        final_test_values = final_test_data[1]


        final_test_utility = torch.zeros(1).to(DEVICE)
        final_test_payment = torch.zeros(1).to(DEVICE)
        final_test_cost = torch.zeros(1).to(DEVICE)

        for num in range(int(final_test_values.shape[0] / bs)):
            choice_id, allocation, utility, payment, allocs, _, _, _, cost= model.test_time_forward(
                torch.tensor(final_test_bids[num * bs:(num + 1) * bs]).to(DEVICE),
                torch.tensor(final_test_values[num * bs:(num + 1) * bs]).to(DEVICE))
            final_test_utility += utility.sum()
            final_test_payment += payment.sum()
            final_test_cost += cost.sum()
        final_test_utility /= final_test_values.shape[0]
        final_test_payment /= final_test_values.shape[0]
        final_test_cost /= final_test_values.shape[0]

        logger.info(
            f"final_test_sp_utility: {final_test_utility},"
            f"final_test_user_utility: {final_test_payment-final_test_cost}," 
            f"final_test_payment: {final_test_payment}," 
            f"final_test_cost: {final_test_cost}")

