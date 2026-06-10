import torch
from torch import nn
import torch.nn.functional as F
from g_transformer.g_trans import GTRANRel_feat


def list_loss(y_pred, y_true, list_num=10, eps=1e-10):
    n_node = y_pred.shape[0]

    ran_num = list_num - 1
    indices = torch.multinomial(torch.ones(n_node), n_node*ran_num, replacement=True).to(y_pred.device)

    list_pred = torch.index_select(y_pred, 0, indices).reshape(n_node, ran_num)
    list_true = torch.index_select(y_true, 0, indices).reshape(n_node, ran_num)

    list_pred = torch.cat([y_pred, list_pred], -1) # [n_node, list_num]
    list_true = torch.cat([y_true, list_true], -1) # [n_node, list_num]

    list_pred = F.softmax(list_pred, -1)
    list_true = F.softmax(list_true, -1)
    list_pred = list_pred + eps
    log_pred = torch.log(list_pred)
    return torch.mean(-torch.sum(list_true * log_pred, dim=1))


# one DJE layer
class DJE_layer(nn.Module):
    def __init__(self,
                in_dim,
                heads_num
                ):
        super(DJE_layer, self).__init__()
        self.in_dim = in_dim
        self.sqrt_d = in_dim ** 0.5
        self.heads_num=heads_num
        self.heads_dim= int(self.in_dim / self.heads_num)
        
        self.w_qs = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim, bias=False),
            nn.ELU(inplace=True)
            )
        self.w_ks = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim, bias=False),
            nn.ELU(inplace=True)
            )
        self.w_vs = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim, bias=False),
            nn.ELU(inplace=True)
            )
        
        self.w_qu = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim, bias=False),
            nn.ELU(inplace=True)
            )
        self.w_ku = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim, bias=False),
            nn.ELU(inplace=True)
            )
        self.w_vu = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim, bias=False),
            nn.ELU(inplace=True)
            )
        
        
        
        self.FFN_s = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim*2),
            nn.ELU(inplace=True),
            nn.Linear(self.in_dim*2, self.in_dim),
            nn.ELU(inplace=True)
        )
        
        self.FFN_u = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim*2),
            nn.ELU(inplace=True),
            nn.Linear(self.in_dim*2, self.in_dim),
            nn.ELU(inplace=True)
        )
        
        self.layer_norm_s_1 = nn.LayerNorm(self.in_dim, eps=1e-6)
        self.layer_norm_s_2 = nn.LayerNorm(self.in_dim, eps=1e-6)
        
        self.layer_norm_u_1 = nn.LayerNorm(self.in_dim, eps=1e-6)
        self.layer_norm_u_2 = nn.LayerNorm(self.in_dim, eps=1e-6)
        
    def forward(self, s, u):
        qs = self.w_qs(s).reshape(s.shape[0],self.heads_num,s.shape[1],self.heads_dim) # [B, heads_num, N, heads_dim]
        ks = self.w_ks(s).reshape(s.shape[0],self.heads_num,s.shape[1],self.heads_dim) # [B, heads_num, N, heads_dim]
        vs = self.w_vs(s).reshape(s.shape[0],self.heads_num,s.shape[1],self.heads_dim) # [B, heads_num, N, heads_dim]
        
        qu = self.w_qu(u).reshape(s.shape[0],self.heads_num,s.shape[1],self.heads_dim) # [B, heads_num, N, heads_dim]
        ku = self.w_ku(u).reshape(s.shape[0],self.heads_num,s.shape[1],self.heads_dim) # [B, heads_num, N, heads_dim]
        vu = self.w_vu(u).reshape(s.shape[0],self.heads_num,s.shape[1],self.heads_dim) # [B, heads_num, N, heads_dim]
        
        ua_s_attn = torch.matmul(qs / self.sqrt_d, ks.transpose(-2,-1)) # [B, heads_num, N, N]
        ua_s_attn = F.softmax(ua_s_attn,dim=-1) # [B, heads_num, N, N]
        ua_s = torch.matmul(ua_s_attn, vs).view(s.shape[0],s.shape[1],s.shape[2]) # [B, N, 2d]
        
        ua_u_attn = torch.matmul(qu / self.sqrt_d, ku.transpose(-2,-1))
        ua_u_attn = F.softmax(ua_u_attn,dim=-1) # [B, heads_num, N, N]
        ua_u = torch.matmul(ua_u_attn, vu).view(u.shape[0],u.shape[1],u.shape[2]) # [B, N, 2d]
        
        s_hat = self.layer_norm_s_1(s + ua_s) # [B, N, 2d]
        s_output = self.layer_norm_s_2(s_hat + self.FFN_s(s_hat)) # [B, N, 2d]
        
        u_hat = self.layer_norm_u_1(u + ua_u) # [B, N, 2d]
        u_output = self.layer_norm_u_2(u_hat + self.FFN_u(u_hat)) # [B, N, 2d]
        
        return s_output, u_output

class DJE(nn.Module):
    def __init__(self,
                args,
                in_dim,
                batch_size,
                ):
        super(DJE, self).__init__()
        self.unc_layers = args.unc_layers
        self.in_dim = in_dim
        self.N=args.uhgt_in_dim
        self.batch_size=batch_size
        self.heads_num=args.uhgt_heads

        # Keep the uncertainty prompts on the module so they follow the model
        # device and are optimized correctly during training.
        self.p_s = nn.Parameter(torch.randn(self.batch_size, self.N, 1))
        self.p_u = nn.Parameter(torch.randn(self.batch_size, self.N, 1))
        
        self.unc_trans_layers = nn.ModuleList()
        
        for l in range(self.unc_layers):
            self.unc_trans_layers.append(DJE_layer(self.in_dim,self.heads_num))
        
        self.fc_s = nn.Linear(self.in_dim*self.N, 1)
        self.fc_u = nn.Linear(self.in_dim*self.N, 1)
        
    def forward(self, q_input, centrality, gamma, beta, REnt):
        q_input = q_input.unsqueeze(1) #[B, 1, 2d]
        p_s = self.p_s[: q_input.shape[0]]
        p_u = self.p_u[: q_input.shape[0]]
        s = torch.bmm(p_s, q_input) # [B, N, 2d]
        u = torch.bmm(p_u, q_input) # [B, N, 2d]

        
        for l in range(self.unc_layers):
            s, u = self.unc_trans_layers[l](s, u)
        
        
        
        s=s.view(q_input.shape[0],-1)
        u=u.view(q_input.shape[0],-1)
        
        s=nn.functional.relu(((centrality * gamma + beta).unsqueeze(-1) + (REnt * (1- gamma) + beta).unsqueeze(-1)) * s)
        u=nn.functional.relu(((centrality * gamma + beta).unsqueeze(-1) + (REnt * (1 - gamma) + beta).unsqueeze(-1)) * u)
        
        x_m=self.fc_s(s)
        x_v=self.fc_u(u)
        
        return x_m, x_v

class Easing(nn.Module):
    def __init__(self,
                 args,
                 g,
                 rel_num,
                 struct_in_dim,
                 content_in_dim,
                 centrality,
                 REnt):
        super(Easing, self).__init__()
        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        self.rel_emb = nn.Embedding(rel_num, args.pred_dim)
        self.struct_gtran = GTRANRel_feat(g, args.num_layers, rel_num, args.pred_dim, struct_in_dim,
                                 args.num_hidden, heads, args.in_drop, args.attn_drop,
                                 args.residual, args.norm, args.edge_mode, self.rel_emb)
        self.content_gtran = GTRANRel_feat(g, args.num_layers, rel_num, args.pred_dim, content_in_dim,
                                 args.num_hidden, heads, args.in_drop, args.attn_drop,
                                 args.residual, args.norm, args.edge_mode, self.rel_emb)

        self.feat_drop = args.feat_drop
        self.h_dim = args.num_hidden*heads[-2]
        self.register_buffer("centrality", centrality.float())
        if torch.is_tensor(REnt):
            rent_value = REnt.float()
        else:
            rent_value = torch.zeros_like(self.centrality)
        self.register_buffer("REnt", rent_value)
        self.gamma = nn.Parameter(torch.FloatTensor(size=(1,)))
        self.beta = nn.Parameter(torch.FloatTensor(size=(1,)))
        nn.init.constant_(self.gamma,args.centrality_gamma)
        nn.init.constant_(self.beta,args.centrality_beta)
        
        self.ut=DJE(args, self.h_dim*2, g.number_of_nodes())

    def forward(self, struct_input, content_input, edge_types):
        if self.feat_drop > 0:
            struct_input = F.dropout(struct_input, self.feat_drop, self.training)
            content_input = F.dropout(content_input, self.feat_drop, self.training)
        
        struct_h = self.struct_gtran(struct_input, edge_types)
        content_h = self.content_gtran(content_input, edge_types)
        
        q = torch.cat((struct_h, content_h), 1)
        
        x_m, x_v = self.ut(q, self.centrality, self.gamma, self.beta, self.REnt)
        
        if self.training:
            return x_m, x_v
        else:
            return x_m, x_v
