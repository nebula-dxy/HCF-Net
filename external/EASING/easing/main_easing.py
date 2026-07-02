import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
import dgl
import os
import sys
import pickle as pk


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)
sys.path.append(os.path.join(rootPath, "utils"))

try:
    from utils.EarlyStopping import EarlyStopping_simple
    from utils.utils import set_random_seed, load_data, rank_evaluate, get_centrality, get_relative_entropy
    from utils.metric import overlap
except ModuleNotFoundError:
    from EarlyStopping import EarlyStopping_simple
    from utils import set_random_seed, load_data, rank_evaluate, get_centrality, get_relative_entropy
    from metric import overlap
from easing.model import list_loss, Easing

def main(args):

    set_random_seed(0)

    ndcg_scores = []
    spearmans = []
    rmses = []
    overlaps = []
    maes=[]
    nrmses=[]

    # set the save path
    save_root = 'results/' + args.dataset + '_EASING/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for cross_id in range(args.cross_num):
        graph_data_path = args.data_path + '/' + args.graph_data
        structure_data_path = args.data_path + '/' + args.structure_data
        semantic_data_path = args.data_path +'/' + args.semantic_data
        split_data_path = args.data_path + f'/{args.dataset}/datasets_split/' + args.split_data
        g, edge_types, _, rel_num, struct_feat, content_feat, train_idx, val_idx, test_idx, train_labels, val_labels, test_labels ,unlabeled_idx, ___= \
            load_data(graph_data_path, structure_data_path, semantic_data_path, args.dataset, args.train_num, split_data_path, args.num_split_idx)
        
        y_mean=torch.mean(train_labels).item()
        y_std=torch.sqrt(torch.var(train_labels)).item()
        if args.gpu >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
        
        
        if args.gpu < 0 or not torch.cuda.is_available():
            cuda = False
        else:
            cuda = True
            g = g.int().to(args.gpu)
            train_labels=train_labels.cuda()
            val_labels=val_labels.cuda()
            test_labels=test_labels.cuda()
            struct_feat = struct_feat.cuda()
            content_feat = content_feat.cuda()
                
        
        if args.REnt_bool == True:
            REnt = get_relative_entropy(g, content_feat)
            if cuda:
                REnt = REnt.cuda()
        else:
            REnt=0

        num_struct_feat = struct_feat.shape[1]
        num_content_feat = content_feat.shape[1]
        n_edges = g.number_of_edges()

        print("""----Data statistics------'
          #Edges %d
          #Unlabeled nodes %d
          #Train samples %d
          #Val samples %d
          #Test samples %d""" %
              (n_edges,
               len(unlabeled_idx),
               len(train_idx),
               len(val_idx),
               len(test_idx)))

        # add self loop
        g = dgl.add_self_loop(g)
        new_edge_types = torch.tensor([rel_num for _ in range(g.number_of_nodes())])
        edge_types = torch.cat([edge_types, new_edge_types], 0)
        rel_num += 1
        n_edges = g.number_of_edges()
        
        
        
        # create models
        loss_fcn = torch.nn.MSELoss()
        model= Easing(args, g, rel_num, num_struct_feat, num_content_feat, get_centrality(g), REnt)
        model_1= Easing(args, g, rel_num, num_struct_feat, num_content_feat, get_centrality(g), REnt)
        
        # save checkpoint
        model_path = save_root + str(cross_id) + '_' + args.save_path
        model_1_path = save_root + str(cross_id) + '_1_' + args.save_path
        if args.early_stop:
            stopper = EarlyStopping_simple(patience=args.patience, save_path=model_path, min_epoch=args.min_epoch)
            stopper_1 = EarlyStopping_simple(patience=args.patience, save_path=model_1_path, min_epoch=args.min_epoch)
        if cuda:
            model_1.cuda()
            model.cuda()
            edge_types = edge_types.cuda()

        # use Adam optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_1 = torch.optim.Adam(
            model_1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        ############################################################################################
        # model training
        ############################################################################################
        dur = []
        for epoch in range(args.epochs):
            model.train()
            model_1.train()
            if epoch >= 3:
                t0 = time.time()
            mean2s_0_stack_ls = []
            mean2s_1_stack_ls = []
            var1s_0_stack_ls = []
            var1s_1_stack_ls = []
            
            torch.set_grad_enabled(True)
            
            ############################################################################################
            # predict values and variance of both labeled data and unlabeled data
            ############################################################################################
            all_output_model_0_pred, var_pred_0 = model(struct_feat, content_feat, edge_types)
            all_output_model_1_pred, var_pred_1 = model_1(struct_feat, content_feat, edge_types)
            
            # labeled data
            mean = all_output_model_0_pred[train_idx].view(-1)
            var = var_pred_0[train_idx].view(-1)
            mean_1 = all_output_model_1_pred[train_idx].view(-1)
            var_1 = var_pred_1[train_idx].view(-1)
            train_labels_logits = ((mean  + mean_1)/2) * y_std + y_mean
            
            # unlabeled data
            all_output_unlb_0_pred_0=all_output_model_0_pred[unlabeled_idx]
            var_unlb_0_pred_0=var_pred_0[unlabeled_idx]
            all_output_unlb_1_pred_0=all_output_model_1_pred[unlabeled_idx]
            var_unlb_1_pred_0=var_pred_1[unlabeled_idx]
            ############################################################################################
            # use unlabeled data
            ############################################################################################
            mean1s_0 = []
            mean2s_0 = []
            var1s_0 = []
            mean1s_1 = []
            mean2s_1 = []
            var1s_1 = []
            unlabeled_idx_in=unlabeled_idx
            # generate pseudo-labels 
            with torch.no_grad():
                for _ in range(args.samp_ssl):
                    mean1_raw_0, var1_raw_0= model(struct_feat, content_feat, edge_types)
                    mean1_raw_1, var1_raw_1= model_1(struct_feat, content_feat, edge_types)
                    mean1_0 = mean1_raw_0[unlabeled_idx_in].view(-1)
                    var1_0 = var1_raw_0[unlabeled_idx_in].view(-1)

                    mean1s_0.append(mean1_0** 2)
                    mean2s_0.append(mean1_0)
                    var1s_0.append(var1_0)

                    mean1_1 = mean1_raw_1[unlabeled_idx_in].view(-1)
                    var1_1 = var1_raw_1[unlabeled_idx_in].view(-1)

                    mean1s_1.append(mean1_1** 2)
                    mean2s_1.append(mean1_1)
                    var1s_1.append(var1_1)
            
            mean2s_0_stack = torch.stack(mean2s_0, dim=1).to("cpu").detach().numpy()
            mean2s_0_stack_ls.append(mean2s_0_stack)
            var1s_0_stack = torch.stack(var1s_0, dim=1).to("cpu").detach().numpy()
            var1s_0_stack_ls.append(var1s_0_stack)

            mean1s_0_ = torch.stack(mean1s_0, dim=0).mean(dim=0)
            mean2s_0_ = torch.stack(mean2s_0, dim=0).mean(dim=0)
            var1s_0_ = torch.stack(var1s_0, dim=0).mean(dim=0)

            mean2s_1_stack = torch.stack(mean2s_1, dim=1).to("cpu").detach().numpy()
            mean2s_1_stack_ls.append(mean2s_1_stack)
            var1s_1_stack = torch.stack(var1s_1, dim=1).to("cpu").detach().numpy()
            var1s_1_stack_ls.append(var1s_1_stack)

            mean1s_1_ = torch.stack(mean1s_1, dim=0).mean(dim=0)
            mean2s_1_ = torch.stack(mean2s_1, dim=0).mean(dim=0)
            var1s_1_ = torch.stack(var1s_1, dim=0).mean(dim=0)


            all_output_unlb_0_pslb = mean2s_0_
            all_output_unlb_1_pslb = mean2s_1_
            
            # calculate heteroscedastic loss of unlabeled data 
            avg_mean01 = (all_output_unlb_0_pslb + all_output_unlb_1_pslb)/2
            avg_var01 = (var1s_0_ + var1s_1_)/2

            loss_mse_cps_0 = ((all_output_unlb_0_pred_0.view(-1) - avg_mean01)**2)
            loss_mse_cps_1 = ((all_output_unlb_1_pred_0.view(-1) - avg_mean01)**2)

            loss_cmb_cps_0 = 0.5 * (torch.mul(torch.exp(-avg_var01), loss_mse_cps_0) + avg_var01 )
            loss_cmb_cps_1 = 0.5 * (torch.mul(torch.exp(-avg_var01), loss_mse_cps_1) + avg_var01 )

            loss_reg_cps0 = loss_cmb_cps_0.mean()
            loss_reg_cps1 = loss_cmb_cps_1.mean()
            
            var_loss_ulb_0 = ((var_unlb_0_pred_0.view(-1) - avg_var01)**2).mean()
            var_loss_ulb_1 = ((var_unlb_1_pred_0.view(-1) - avg_var01)**2).mean()
            
            loss_reg_cps = (loss_reg_cps0 + loss_reg_cps1) + (var_loss_ulb_0 + var_loss_ulb_1)
            
            ############################################################################################
            # use labeled data
            ############################################################################################
            loss_mse = (mean - (train_labels - y_mean) / y_std) ** 2
            loss_mse_1 = (mean_1 - (train_labels - y_mean) / y_std) ** 2
            list_loss_labels=list_loss(train_labels_logits.unsqueeze(1),train_labels.unsqueeze(1),args.list_num)
            
            # calculate heteroscedastic loss of labeled data
            loss1 = torch.mul(torch.exp(-(var + var_1) / 2), loss_mse)
            loss2 = (var + var_1) / 2
            loss = .5 * (loss1 + loss2)
            loss_reg_0 = loss.mean()
            
            loss1_1 = torch.mul(torch.exp(-(var + var_1) / 2), loss_mse_1)
            loss2_1 = (var + var_1) / 2
            loss_1 = .5 * (loss1_1 + loss2_1)
            loss_reg_1 = loss_1.mean()
            
            loss_reg = (loss_reg_0 + loss_reg_1)
            
            # calculate final loss
            loss = loss_reg + args.w_ulb * loss_reg_cps + ((var_1 - var) ** 2).mean() + list_loss_labels * args.loss_beta
            
            optimizer.zero_grad()
            optimizer_1.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_1.step()
            
            if epoch >= 3:
                dur.append(time.time() - t0)
            
            
            ############################################################################################
            # model evaluation
            ############################################################################################
            model.eval()
            model_1.eval()
            with torch.no_grad():
                mean2s_v0=[]
                mean2s_v1=[]
                for _ in range(args.samp_fq):
                    val_mean,val_var = model(struct_feat, content_feat, edge_types)
                    val_mean_1,val_var_1 = model_1(struct_feat, content_feat, edge_types)
                    
                    mean1_v0 = val_mean.view(-1)
                    mean2s_v0.append(mean1_v0)
                    
                    mean1_v1 = val_mean_1.view(-1)
                    mean2s_v1.append(mean1_v1)
                    
                    
                mean1s_v1_ = torch.stack(mean2s_v0, dim=0).mean(dim=0)
                mean2s_v1_ = torch.stack(mean2s_v1, dim=0).mean(dim=0)
                
                val_logits = ((mean1s_v1_+mean2s_v1_) / 2) * y_std + y_mean
                print(val_logits)

                _, val_ndcg, val_spm,_= rank_evaluate(val_logits[val_idx], val_labels.unsqueeze(-1), 100, loss_fcn, True)
                test_loss, test_ndcg, test_spm,test_mae = rank_evaluate(val_logits[test_idx], test_labels.unsqueeze(-1), 100, loss_fcn, True)

            if args.early_stop:
                if args.spm:
                    stop = stopper.step(val_spm, epoch, model)
                    stop_1 = stopper_1.step(val_spm, epoch, model_1)
                else:
                    stop = stopper.step(val_ndcg, epoch, model)
                    stop_1 = stopper_1.step(val_ndcg, epoch, model_1)
                if stop and stop_1:
                    print('best epoch :', stopper.best_epoch)
                    print('best epoch :', stopper_1.best_epoch)
                    break

            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Loss_reg {:.4f} | test_mae {:.4f} | test_rmse {:.4f} |"
                  " ValSPM {:.4f} | ValNDCG {:.4f} | TestSPM {:.4f} | TestNDCG {:.4f}".
                  format(epoch, np.mean(dur), loss.item(), loss_reg.item(), float(test_mae), torch.sqrt(test_loss).item(),
                         val_spm, val_ndcg, test_spm, test_ndcg))

        print()
        if args.early_stop:
            model.load_state_dict(torch.load(model_path))
            model_1.load_state_dict(torch.load(model_1_path))


        ############################################################################################
        # model test
        ############################################################################################
        model.eval()
        model_1.eval()
        with torch.no_grad():
            mean2s_t0=[]
            mean2s_t1=[]
            for _ in range(args.samp_fq):
                test_mean,test_var = model(struct_feat, content_feat, edge_types)
                test_mean_1,test_var_1 = model_1(struct_feat, content_feat, edge_types)

                mean1_t0 = test_mean.view(-1)
                mean2s_t0.append(mean1_t0)

                mean1_t1 = test_mean_1.view(-1)
                mean2s_t1.append(mean1_t1)

                    
            mean1s_t1_ = torch.stack(mean2s_t0, dim=0).mean(dim=0)
            mean2s_t1_ = torch.stack(mean2s_t1, dim=0).mean(dim=0)

            test_logits=((mean1s_t1_+mean2s_t1_)/2)* y_std + y_mean
            test_loss, test_ndcg, test_spearman,test_mae = \
                rank_evaluate(test_logits[test_idx], test_labels.unsqueeze(-1), 100, loss_fcn, spearman=True)
            test_overlap = overlap(test_labels, test_logits[test_idx], 100)
            nrmse=(test_loss / (max(test_labels) - min(test_labels))).item()
            print("Test NDCG {:.4f} | Test RMSE {:.4f} | Test Spearman {:.4f} | Test Overlap {:.4f}| Test MAE {:.4f}| Test NRMSE {:.4f}".
                  format(test_ndcg, torch.sqrt(test_loss).item(), test_spearman, test_overlap,test_mae, nrmse))
        pred_path = save_root + str(cross_id) + '_' + args.save_path.replace('checkpoint.pt', 'pred.npy')
        np.save(pred_path, test_logits.detach().cpu().numpy())
            
        ndcg_scores.append(test_ndcg)
        spearmans.append(test_spearman)
        maes.append(float(test_mae))
        rmses.append(torch.sqrt(test_loss).item())
        nrmse=(torch.sqrt(test_loss) / (max(test_labels) - min(test_labels))).item()
        nrmses.append(nrmse)
        overlaps.append(test_overlap)
        
        
    print()
    ndcg_scores = np.array(ndcg_scores)
    print('ndcg: ', ndcg_scores, ndcg_scores.mean(), np.std(ndcg_scores))

    spearmans = np.array(spearmans)
    print('spearmans: ', spearmans, spearmans.mean(), np.std(spearmans))

    rmses = np.array(rmses)
    print('RMSE: ', rmses, rmses.mean(), np.std(rmses))
    
    nrmses = np.array(nrmses)
    print('NRMSE: ', nrmses, nrmses.mean(), np.std(nrmses))
    
    maes = np.array(maes)
    print('MAE: ', maes, maes.mean(), np.std(maes))

    overlaps = np.array(overlaps)
    print(overlaps, overlaps.mean(), np.std(overlaps))
    
    safe_name = f"{args.dataset}_result_one_label_pred_{args.train_num}_1_1_w_ulb_{args.w_ulb}_loss_beta_{args.loss_beta}_T_{args.samp_ssl}_unc_layers_{args.unc_layers}_cen_{args.centrality_gamma}.txt"
    with open(safe_name, 'w') as f:
        f.write(f'{args.train_num}:1:1\n'
                f'{len(train_idx)}\n'
                +f'ndcg: {ndcg_scores}, {ndcg_scores.mean()}, {np.std(ndcg_scores)}\n'
                +f'spearmans: {spearmans}, {spearmans.mean()}, {np.std(spearmans)}\n'
                +f'RMSE: {rmses}, {rmses.mean()}, {np.std(rmses)}\n'
                +f'NRMSE: {nrmses}, {nrmses.mean()}, {np.std(nrmses)}\n'
                +f'MAE: {maes}, {maes.mean()}, {np.std(maes)}\n'
                +f'{overlaps}, {overlaps.mean()}, {np.std(overlaps)}\n'
                )

    results = {'ndcg': ndcg_scores,
               'spearman': spearmans,
               'rmse': rmses,
               'overlap': overlaps,
               'mae': maes,
               'nrmse': nrmses,
               'pred_path_pattern': save_root + '{cross_id}_' + args.save_path.replace('checkpoint.pt', 'pred.npy'),
               'args': vars(args)
               }
    

    result_path = save_root + args.save_path.replace('checkpoint.pt', '') + 'result.pk'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    pk.dump(results, open(result_path, 'wb'))
    pk.dump(results, open(result_path, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EASING')
    parser.add_argument("--dataset", type=str, default='FB15k',
                        help="The input dataset. Can be FB15K, TMDB5K, IMDB")
    parser.add_argument("--data_path", type=str, help="path of your new dataset")

    parser.add_argument("--graph_data", type=str, default='FB15K/fb15k_rel.pk',
                        help="path of xx_rel in your new dataset, such as FB15K/fb15k_rel.pk")
    parser.add_argument("--semantic_data", type=str, default='FB15K/fb_lang.pk',
                        help="path of xx_lang in your new dataset, such as FB15/fb_lang.pk")
    parser.add_argument("--structure_data", type=str, default='IMDB/imdb_s_node2vec_new_150000.pk',
                        help="only used for IMDB")
    parser.add_argument("--split_data", type=str, default='idx_1000',
                        help="path of split dataset, your_split_data_path/idx_(split num)")
    parser.add_argument('--num_split_idx', type=int, default=1000,
                        help="number of split index, same as the number of files in the split dataset")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which gpu to use. Set -1 to use CPU.")
    parser.add_argument("--cross-num", type=int, default=5,
                        help="number of cross validation")
    parser.add_argument("--epochs", type=int, default=10000,
                        help="number of training epochs")
    parser.add_argument('--min-epoch', type=int, default=-1,
                        help='the least epoch for training, avoiding stopping at the start time')
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=4,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_false",
                        help="use residual connection")
    parser.add_argument("--feat-drop", type=float, default=0.)
    parser.add_argument("--in-drop", type=float, default=.3,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.3,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true',
                        help="indicates whether to use early stop or not")
    parser.add_argument('--patience', type=int, default=500,
                        help="indicates when to early stop")
    parser.add_argument('--pred-dim', type=int, default=10,
                        help="the size of predicate embedding vector")
    parser.add_argument('--save-path', type=str, default='checkpoint_fb15k.pt',
                        help='the path to save the best model')

    parser.add_argument('--norm', action="store_false")
    parser.add_argument('--edge-mode', type=str, default='MUL')
    parser.add_argument('--spm', action="store_false")
    parser.add_argument('--list-num', type=int, default=100)
    parser.add_argument("--train_num", type=float, default=1.0,
                        help="percentage of labeled data. 1.0 means 10% nodes were labeled") 
    parser.add_argument('--samp_ssl', type=int, default=5)
    parser.add_argument('--w_ulb', type=float, default=1.0)
    parser.add_argument('--samp_fq', type=int, default=5)
    parser.add_argument('--loss_beta', type=float, default=0.0)
    parser.add_argument('--centrality_gamma', type=float, default=0.9)
    parser.add_argument('--centrality_beta', type=float, default=0.0)
    
    parser.add_argument('--REnt_bool', action="store_false")
    parser.add_argument("--unc_layers", type=int, default=2,
                        help="number of layers in uncertainty prediction transformer")
    parser.add_argument("--uhgt_in_dim", type=int, default=1,
                        help="uhgt of in dimension, called N")
    parser.add_argument("--uhgt_heads", type=int, default=8,
                        help="uhgt of in dimension, called N")

    args = parser.parse_args()
    print(args)
    main(args)
    print(f"{args.train_num}:1:1")
    
    
