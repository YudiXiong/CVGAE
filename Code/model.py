"""
Author: Yudi Xiong
Google Scholar: https://scholar.google.com/citations?user=LY4PK9EAAAAJ
ORCID: https://orcid.org/0009-0001-3005-8225
Date: April, 2024
"""

import torch
from torch import nn
import numpy as np
import os
import sys
import torch.nn.functional as F
import random
import pandas as pd

torch.autograd.set_detect_anomaly(True)

class CVGAE(nn.Module):
    def __init__(self, data_config, args, pretrain_data, device):
        super(CVGAE, self).__init__()
        self.device = device
        self.to(self.device)

        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.initial_type = args.initial_type
        self.pretrain_data = pretrain_data
        self.fuse_type_in = args.fuse_type_in

        self.n_users = data_config['n_users']
        self.n_items_s = data_config['n_items_s']
        self.n_items_t = data_config['n_items_t']

        self.n_fold = 1
        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = eval(args.node_dropout)
        self.mess_dropout = eval(args.mess_dropout)

        self.norm_adj_s = data_config['norm_adj_s']
        self.norm_adj_t = data_config['norm_adj_t']
        self.n_nonzero_elems_s = self.norm_adj_s.count_nonzero()
        self.n_nonzero_elems_t = self.norm_adj_t.count_nonzero()

        self.domain_laplace = torch.FloatTensor(data_config['domain_adj']).to(self.device)
        self.connect_way = args.connect_type
        self.layer_fun = args.layer_fun

        self.lr = args.lr
        self.n_interaction = args.n_interaction

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose

        self.lambda_s = eval(args.lambda_s)
        self.lambda_t = eval(args.lambda_t)

        self.weight_source, self.weight_target = eval(args.weight_loss)[:]

        self.loss_fun = nn.BCEWithLogitsLoss()

        self.dropout = nn.Dropout(p=self.mess_dropout[0])

        self.beta = args.beta

        self.weights_source = self._init_weights('source', self.n_items_s, None).to(self.device)
        self.weights_target = self._init_weights('target', self.n_items_t, None).to(self.device)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.W_i_source = torch.randn(self.emb_dim, 2 * self.emb_dim).to(self.device)
        self.W_i_target = torch.randn(self.emb_dim, 2 * self.emb_dim).to(self.device)

        self.pro4Vae_c = nn.Linear((self.n_layers+1) * self.emb_dim, self.emb_dim)
        self.pro4Vae_c = self.pro4Vae_c.to(device)
        self.Relu1_c = nn.LeakyReLU()

        self.pro4Vae_s = nn.Linear((self.n_layers+1) * self.emb_dim, self.emb_dim)
        self.pro4Vae_s = self.pro4Vae_s.to(device)
        self.Relu1_s = nn.LeakyReLU()

        self.pro4Vae_t = nn.Linear((self.n_layers+1) * self.emb_dim, self.emb_dim)
        self.pro4Vae_t = self.pro4Vae_t.to(device)
        self.Relu1_t = nn.LeakyReLU()

        self.pro4Vae_s_i = nn.Linear((self.n_layers+1) * self.emb_dim, self.emb_dim)
        self.pro4Vae_s_i = self.pro4Vae_s_i.to(device)
        self.Relu1_i = nn.LeakyReLU()

        self.pro4Vae_t_i = nn.Linear((self.n_layers+1) * self.emb_dim, self.emb_dim)
        self.pro4Vae_t_i = self.pro4Vae_t_i.to(device)
        self.Relu2_i = nn.LeakyReLU()

        self.margin = 0.5
        self.mu_uweight = args.mu_uweight

        self.csv_file_path=args.csv_file_path
        self.fusionbeta=args.fusionbeta

        self.csv_file_path2 = '../attack/user_camouflaged_embeddings_gender_128_0.55.csv'
        self.df = pd.read_csv(self.csv_file_path2, header=None)
        self.df_tensor = torch.tensor(self.df.values, dtype=self.weights_source["user_embedding"].dtype)
        self.df_tensor = self.df_tensor.to(self.weights_source["user_embedding"].device)

        self.df_t = pd.read_csv(self.csv_file_path, header=None)
        self.df_tensor_t = torch.tensor(self.df_t.values, dtype=self.weights_source["user_embedding"].dtype)
        self.df_tensor_t = self.df_tensor_t.to(self.weights_source["user_embedding"].device)

    def _init_weights(self, name_scope, n_items, user_embedding):
        initializer = nn.init.xavier_uniform_
        weight_dict = nn.ParameterDict()
        if self.pretrain_data == None:
            if user_embedding == None:
                weight_dict['user_embedding'] = nn.Parameter(initializer(torch.empty(self.n_users, self.emb_dim)))
                weight_dict['item_embedding'] = nn.Parameter(initializer(torch.empty(n_items, self.emb_dim)))
            else:
                weight_dict['user_embedding'] = nn.Parameter(initializer(user_embedding))
                weight_dict['item_embedding'] = nn.Parameter(initializer(torch.empty(n_items, self.emb_dim)))

        return weight_dict

    def _split_A_hat(self, X, n_items):
        A_fold_hat = []

        fold_len = (self.n_users + n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X, n_items):
        A_fold_hat = []

        fold_len = (self.n_users + n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + n_items
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_embed(self, weights_s, weights_t, norm_adj_s, norm_adj_t):
        global all_embeddings_s, all_embeddings_t

        def one_graph_layer_gcf(A_fold_hat, ego_embeddings, weights, k):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(torch.sparse.mm(A_fold_hat[f].to(self.device), ego_embeddings))

            side_embeddings = torch.concat(temp_embed, 0)

            ego_embeddings = side_embeddings + torch.mul(ego_embeddings, side_embeddings)

            ego_embeddings = self.dropout(ego_embeddings)

            return ego_embeddings

        if self.node_dropout_flag:

            A_fold_hat_s = self._split_A_hat_node_dropout(norm_adj_s, self.n_items_s)
            A_fold_hat_t = self._split_A_hat_node_dropout(norm_adj_t, self.n_items_t)
            print("启用node dropout")
        else:
            A_fold_hat_s = self._split_A_hat(norm_adj_s, self.n_items_s)
            A_fold_hat_t = self._split_A_hat(norm_adj_t, self.n_items_t)

        ego_embeddings_s = torch.concat([weights_s['user_embedding'], weights_s['item_embedding']], dim=0)
        ego_embeddings_t = torch.concat([weights_t['user_embedding'], weights_t['item_embedding']], dim=0)

        if self.connect_way == 'concat':
            all_embeddings_s = [ego_embeddings_s]
            all_embeddings_t = [ego_embeddings_t]
        elif self.connect_way == 'mean':
            all_embeddings_s = ego_embeddings_s
            all_embeddings_t = ego_embeddings_t

        for k in range(0, self.n_layers):
            if self.layer_fun == 'gcf':
                ego_embeddings_s = one_graph_layer_gcf(A_fold_hat_s, ego_embeddings_s, weights_s, k)
                ego_embeddings_t = one_graph_layer_gcf(A_fold_hat_t, ego_embeddings_t, weights_t, k)
            if k >= self.n_layers - self.n_interaction and self.n_interaction > 0:
                if self.fuse_type_in == 'la2add':
                    ego_embeddings_s, ego_embeddings_t = self.s_t_la2add_layer(ego_embeddings_s, ego_embeddings_t)

            norm_embeddings_s = F.normalize(ego_embeddings_s, p=2, dim=1)
            norm_embeddings_t = F.normalize(ego_embeddings_t, p=2, dim=1)

            if self.connect_way == 'concat':
                all_embeddings_s += [norm_embeddings_s]
                all_embeddings_t += [norm_embeddings_t]
            elif self.connect_way == 'mean':
                all_embeddings_s += norm_embeddings_s
                all_embeddings_t += norm_embeddings_t
        if self.connect_way == 'concat':
            all_embeddings_s = torch.concat(all_embeddings_s, 1)
            all_embeddings_t = torch.concat(all_embeddings_t, 1)
        elif self.connect_way == 'mean':
            all_embeddings_s = all_embeddings_s / (self.n_layers + 1)
            all_embeddings_t = all_embeddings_t / (self.n_layers + 1)

        u_g_embeddings_s, i_g_embeddings_s = torch.split(all_embeddings_s, [self.n_users, self.n_items_s], 0)
        u_g_embeddings_t, i_g_embeddings_t = torch.split(all_embeddings_t, [self.n_users, self.n_items_t], 0)

        return u_g_embeddings_s, i_g_embeddings_s, u_g_embeddings_t, i_g_embeddings_t

    def protect_privacy_s(self, user_s_em):
        user_s_noise = torch.nn.Parameter((1-self.fusionbeta) * user_s_em.data + self.fusionbeta * self.df_tensor)
        return user_s_noise

    def protect_privacy_t(self, user_s_em):
        user_s_noise = torch.nn.Parameter((1-0.2) * user_s_em.data + 0.2 * self.df_tensor_t)
        return user_s_noise

    '''
    CVGAE(-C) (a reduced version without privacy protection)
    If you need to run CVGAE, please comment out the current def s_t_la2add_layer corresponding to CVGAE(-C), 
    and instead, uncomment the def s_t_la2add_layer specific to CVGAE. 
    This way, the appropriate function for CVGAE will be used.
    '''
    def s_t_la2add_layer(self, ego_embeddings_s, ego_embeddings_t):
        users_s, items_s = torch.split(ego_embeddings_s, [self.n_users, self.n_items_s])
        users_t, items_t = torch.split(ego_embeddings_t, [self.n_users, self.n_items_t])

        l_s = self.domain_laplace[:, 0]
        l_t = self.domain_laplace[:, 1]

        users_comm = torch.mul(l_s, users_s.T).T + torch.mul(l_t, users_t.T).T

        users_s_special = self.lambda_s * users_s + (1 - self.lambda_s) * users_t
        users_t_special = (1 - self.lambda_t) * users_s + self.lambda_t * users_t

        users_s_transfer = (1 / 2) * (users_comm + users_s_special)
        users_t_transfer = (1 / 2) * (users_comm + users_t_special)

        embs_s = torch.cat([users_s_transfer, items_s])
        embs_t = torch.cat([users_t_transfer, items_t])

        return embs_s, embs_t

    '''
    CVGAE (the complete version with privacy protection)
    '''
    # def s_t_la2add_layer(self, ego_embeddings_s, ego_embeddings_t):
    #     users_s, items_s = torch.split(ego_embeddings_s, [self.n_users, self.n_items_s])
    #     users_t, items_t = torch.split(ego_embeddings_t, [self.n_users, self.n_items_t])
    #
    #     l_s = self.domain_laplace[:, 0]
    #     l_t = self.domain_laplace[:, 1]
    #
    #     users_s_noise = self.protect_privacy_s(users_s.clone())
    #     users_t_noise = self.protect_privacy_t(users_t.clone())
    #
    #     users_comm_noice_t = torch.mul(l_s, users_s_noise.T).T + torch.mul(l_t, users_t.T).T
    #     users_comm_noice_s = torch.mul(l_s, users_s.T).T + torch.mul(l_t, users_t_noise.T).T
    #
    #     users_s_special = self.lambda_s * users_s + (1 - self.lambda_s) * users_t_noise
    #     users_t_special = (1 - self.lambda_t) * users_s_noise + self.lambda_t * users_t
    #
    #     users_s_transfer = (1 / 2) * (users_comm_noice_s + users_s_special)
    #     users_t_transfer = (1 / 2) * (users_comm_noice_t + users_t_special)
    #
    #     embs_s = torch.cat([users_s_transfer, items_s])
    #     embs_t = torch.cat([users_t_transfer, items_t])
    #
    #     return embs_s, embs_t

    def get_scores(self, users, pos_items):
        scores = torch.sum(torch.mul(users, pos_items), dim=1)
        return scores

    def create_cross_loss(self, users, pos_items, labels):
        regularizer = (torch.norm(users, p=2) ** 2
                       + torch.norm(pos_items, p=2) ** 2) / 2
        regularizer = regularizer / self.batch_size

        scores = torch.sum(torch.mul(users, pos_items), dim=1)
        mf_loss = self.loss_fun(input=scores, target=labels)

        emb_loss = self.decay * regularizer

        reg_loss = 0.0

        return mf_loss, emb_loss, reg_loss


    def forward(self, users_s, items_s, users_t, items_t, run_flag):
        ua_embeddings_s, ia_embeddings_s, ua_embeddings_t, ia_embeddings_t = self._create_embed(
            self.weights_source,
            self.weights_target,
            self.norm_adj_s,
            self.norm_adj_t)

        (z_u_source, z_i_source, z_u_target, z_i_target,
        mu_i_source, logsig_i_source, mu_i_target, logsig_i_target,
        mu_u_source, logsig_u_source, mu_u_target, logsig_u_target) = self.VAE_embed(ua_embeddings_s, ia_embeddings_s, ua_embeddings_t, ia_embeddings_t, run_flag)

        z_u_source_4d = torch.cat([z_u_source] * (self.n_layers + 1), dim=1)
        z_i_source_4d = torch.cat([z_i_source] * (self.n_layers + 1), dim=1)
        z_u_target_4d = torch.cat([z_u_target] * (self.n_layers + 1), dim=1)
        z_i_target_4d = torch.cat([z_i_target] * (self.n_layers + 1), dim=1)

        users_embddings_s = z_u_source_4d[users_s]*self.beta+ua_embeddings_s[users_s]
        items_embddings_s = z_i_source_4d[items_s]*self.beta+ia_embeddings_s[items_s]
        users_embddings_t = z_u_target_4d[users_t]*self.beta+ua_embeddings_t[users_t]
        items_embddings_t = z_i_target_4d[items_t]*self.beta+ia_embeddings_t[items_t]

        return users_embddings_s, items_embddings_s, users_embddings_t, items_embddings_t, z_u_source, z_i_source, z_u_target, z_i_target,\
        mu_i_source, logsig_i_source, mu_i_target, logsig_i_target,\
        mu_u_source, logsig_u_source, mu_u_target, logsig_u_target


    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        i = torch.stack([row, col])
        v = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(i, v, coo.shape)


    def _dropout_sparse(self, x, rate, noise_shape):
        random_tensor = 1 - rate

        random_tensor += torch.rand(noise_shape).to(x.device)

        dropout_mask = torch.floor(random_tensor).type(torch.bool)

        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))


    def VAE_embed(self, ua_embeddings_s, ia_embeddings_s, ua_embeddings_t, ia_embeddings_t, run_flag):
        mu_u_source = self.Relu1_c(self.pro4Vae_c(ua_embeddings_s + self.mu_uweight * ua_embeddings_t))
        mu_i_source = self.Relu1_i(self.pro4Vae_s_i(ia_embeddings_s))
        mu_u_target = self.Relu1_c(self.pro4Vae_c(ua_embeddings_t + self.mu_uweight * ua_embeddings_s))
        mu_i_target = self.Relu2_i(self.pro4Vae_t_i(ia_embeddings_t))

        W_i_source = torch.mm(mu_i_source, self.W_i_source)
        mu_i_source = W_i_source[:, :self.emb_dim].to(self.device)
        logsig_i_source = W_i_source[:, self.emb_dim:2*self.emb_dim].to(self.device)
        std_i_source = torch.exp(0.5 * logsig_i_source).to(self.device)
        epsilon_i_source = torch.randn_like(std_i_source).to(self.device)
        z_i_source = mu_i_source + epsilon_i_source * std_i_source * run_flag

        W_i_target = torch.mm(mu_i_target, self.W_i_target).to(self.device)
        mu_i_target = W_i_target[:, :self.emb_dim].to(self.device)
        logsig_i_target = W_i_target[:, self.emb_dim:2*self.emb_dim].to(self.device)
        std_i_target = torch.exp(0.5 * logsig_i_target).to(self.device)
        epsilon_i_target = torch.randn_like(std_i_target).to(self.device)
        z_i_target = mu_i_target + epsilon_i_target * std_i_target * run_flag

        logsig_u_target =  self.Relu1_s(self.pro4Vae_s(ua_embeddings_s))
        std_u_target = torch.exp(0.5 * logsig_u_target).to(self.device)
        epsilon_u_target = torch.randn_like(std_u_target).to(self.device)
        z_u_target = mu_u_target + epsilon_u_target * std_u_target * run_flag
        z_u_target = z_u_target.to(self.device)

        logsig_u_source = self.Relu1_t(self.pro4Vae_t(ua_embeddings_t))
        std_u_source = torch.exp(0.5 * logsig_u_source).to(self.device)
        epsilon_u_source = torch.randn_like(std_u_source).to(self.device)
        z_u_source = mu_u_source + epsilon_u_source * std_u_source * run_flag
        z_u_source = z_u_source.to(self.device)

        return z_u_source, z_i_source, z_u_target, z_i_target, mu_i_source, logsig_i_source, mu_i_target, logsig_i_target, mu_u_source, logsig_u_source, mu_u_target, logsig_u_target



    def create_kl_loss(self,z_u_source, z_i_source, z_u_target, z_i_target, mu_i_source, logsig_i_source, mu_i_target, logsig_i_target, mu_u_source, logsig_u_source, mu_u_target, logsig_u_target):
        KL_u_source = (0.5 / self.n_users) * torch.mean(
            torch.sum((-logsig_u_source + torch.exp(logsig_u_source) + mu_u_source ** 2 - 1), dim=1)
        )

        KL_i_source = (0.5 / self.n_items_s) * torch.mean(
            torch.sum((-logsig_i_source + torch.exp(logsig_i_source) + mu_i_source ** 2 - 1), dim=1)
        )

        KL_u_target = (0.5 / self.n_users) * torch.mean(
            torch.sum((-logsig_u_target + torch.exp(logsig_u_target) + mu_u_target ** 2 - 1), dim=1)
        )

        KL_i_target = (0.5 / self.n_items_t) * torch.mean(
            torch.sum((-logsig_i_target + torch.exp(logsig_i_target) + mu_i_target ** 2 - 1), dim=1)
        )

        KL_source = KL_u_source + KL_i_source
        KL_target = KL_u_target + KL_i_target

        return KL_source, KL_target

