from classifier.model import BertED
from classifier.exemplars import Exemplars
from utils.dataloader import (
    collect_dataset, collect_exemplar_dataset, 
    collect_sldataset, collect_from_json, 
    MAVEN_Dataset,
    DescriptionDataset, collect_eval_sldataset
)
from utils.computeLoss import compute_CLLoss, CrossEntropyLossWithWeight
from utils.tools import contrastive_loss_des, find_negative_labels, collate_description, balance_zero_with_nonzero
from utils.calcs import Calculator

import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, time, sys, json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from torch.nn.functional import normalize
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from copy import deepcopy
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizerFast
import wandb
from loguru import logger
from tqdm.auto import tqdm

def train(local_rank, args):    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Configure logging
    os.makedirs(args.log_dir, exist_ok=True)
    # --- delete default handle ---
    logger.remove()
    # --- add handle ---
    date_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Add timestamp
    args.run_name = f"{args.dataset}_{args.task_num}_{args.shot_num}_{args.class_num}_{args.epochs}_{args.task_ep_time}_{args.seed}_{args.alpha_ce}_{timestamp}"

    
    save_path = os.path.join(args.save_output, args.dataset, str(args.shot_num))
    if os.path.exists(save_path) and os.path.isdir(save_path):
        print(f"✅ Thư mục '{save_path}' tồn tại.")
    else:
        print(f"❌ Thư mục '{save_path}' không tồn tại. Đang tạo mới...")
        os.makedirs(save_path, exist_ok=True)
        print(f"✅ Đã tạo thư mục '{save_path}'.")
    log_file_path = os.path.join(save_path, f"{args.run_name}.txt")
    
    logger.add(
        log_file_path,
        rotation="1 MB", 
        retention="10 days",
        enqueue=True,
        level="INFO"
    )
    
    logger.level("CRITICAL", color="<bg red><white>")
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        level="DEBUG",
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{file: >18}: {line: <4}</cyan> - <level>{message}</level>",
    )
    
    # set the random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # set device, whether to use cuda or cpu
    device = torch.device(args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Thêm timestamp
    args.run_name = f"{args.dataset}_{args.task_num}_{args.shot_num}_{args.epochs}_{args.task_ep_time}_{args.distill}_{args.alpha_ce}_{timestamp}"

    # get streams from json file and permute them in pre-defined order
    # PERM = PERM_5 if args.task_num == 5 else PERM_10
    
    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.project_name,
            name = args.run_name,

            # track hyperparameters and run metadata
            config=args.__dict__,
            reinit=True
        )
    else:
        wandb.init(
            project=args.project_name,
            name = args.run_name,
            mode="disabled"
        )
    
    # Load data
    streams, _ = collect_from_json(args.dataset, args.data_root, 'stream', args)
    # streams = [streams[l] for l in PERM[int(args.perm_id)]] # permute the stream
    label2idx = {0:0}
    idx2label = {}
    
    for st in streams:
        for lb in st:
            if lb not in label2idx:
                label2idx[lb] = len(label2idx)
    
    for key, value in label2idx.items():
        idx2label[value] = key
    
    streams_indexed = [[label2idx[l] for l in st] for st in streams]
    
    if args.backbone_path != "":
        model = BertED(args, args.backbone_path) # define model
    else:
        model = BertED(args) # define model
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay, eps=args.adamw_eps, betas=(0.9, 0.999)) #TODO: Hyper parameters
    
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gammalr) # TODO: Hyper parameters
                  
    if args.parallel == 'DDP':
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", rank=local_rank, world_size=args.world_size)
        # device_id = [0, 1, 2, 3, 4, 5, 6, 7]
        model = DDP(model, device_ids= [local_rank], find_unused_parameters=True)
    elif args.parallel == 'DP':
        os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3, 4, 5, 6, 7' 
        model = nn.DataParallel(model, device_ids=[int(it) for it in args.device_ids.split(" ")])

    criterion_ce = nn.CrossEntropyLoss()
    criterion_fd = nn.CosineEmbeddingLoss()
    all_labels = []
    all_labels = list(set([t for stream in streams_indexed for t in stream if t not in all_labels]))
    task_idx = [i for i in range(len(streams_indexed))]
    labels = all_labels.copy()

    # training process
    learned_types = [0]
    prev_learned_types = [0]
    dev_scores_ls = []
    
    exemplars = Exemplars(args) # TODO: 
    if args.cresume:
        logger.info(f"Resuming from {args.cresume}")
        state_dict = torch.load(args.cresume)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        task_idx = task_idx[state_dict['stage']:]
        # TODO: test use
        labels = state_dict['labels']
        learned_types = state_dict['learned_types']
        prev_learned_types = state_dict['prev_learned_types']
        
    
    e_pth = "./outputs/early_stop/" + args.log_name + ".pth"
    os.makedirs(os.path.dirname(e_pth), exist_ok=True)
    
    for stage in task_idx:
        # if stage > 0:
        #     break
        logger.info(f"==================== Stage {stage} ====================")
        
        # stage = 1 # TODO: test use
        # exemplars = Exemplars() # TODO: test use
        if args.single_label:
            stream_dataset = collect_sldataset(args.dataset, args.data_root, 'train', label2idx, stage, streams[stage], args)
        else:
            stream_dataset = collect_dataset(args.dataset, args.data_root, 'train', label2idx, stage, [i for item in streams[stage:] for i in item], args)
        if args.parallel == 'DDP':
            stream_sampler = DistributedSampler(stream_dataset, shuffle=True)
            org_loader = DataLoader(
                dataset=stream_dataset,
                sampler=stream_sampler,
                batch_size=args.batch_size,
                # batch_size=args.shot_num + int(args.class_num / args.shot_num),
                collate_fn= lambda x:x)
        else:
            org_loader = DataLoader(
                dataset=stream_dataset,
                shuffle=True,
                batch_size=args.batch_size,
                # batch_size=args.shot_num + int(args.class_num / args.shot_num),
                collate_fn= lambda x:x)
            
        stage_loader = org_loader
        if stage > 0:
            if args.early_stop and no_better >= args.patience:
                logger.info("Early stopping finished, loading stage: " + str(stage))
                model.load_state_dict(torch.load(e_pth))
            prev_model = deepcopy(model) # TODO:test use
            for item in streams_indexed[stage - 1]:
                if not item in prev_learned_types:
                    prev_learned_types.append(item)
            # TODO: test use
            # prev_model = deepcopy(model) # TODO: How does optimizer distinguish deep copy parameters
            # exclude_none_labels = [t for t in streams_indexed[stage - 1] if t != 0]
            logger.info(f'Loading train instances without negative instances for stage {stage}')
            
            exemplar_dataset = collect_exemplar_dataset(args.dataset, args.data_root, 'train', label2idx, stage-1, streams[stage-1], args)
            exemplar_loader = DataLoader(
                dataset=exemplar_dataset,
                batch_size=64,
                shuffle=True,
                collate_fn=lambda x:x)

            exemplars.set_exemplars(prev_model, exemplar_loader, len(learned_types), device)
            # if not args.replay:
            if not args.no_replay:
                stage_loader = exemplars.build_stage_loader(stream_dataset)
            # else:
            #     e_loader = list(exemplars.build_stage_loader(MAVEN_Dataset([], [], [], [])))
            if args.rep_aug != "none":
                e_loader = exemplars.build_stage_loader(MAVEN_Dataset([], [], [], [], []))
            # prev_model.to(args.device)   # TODO: test use

        for item in streams_indexed[stage]:
            if not item in learned_types:
                learned_types.append(item)
        logger.info(f'Learned types: {learned_types}')
        logger.info(f'Previous learned types: {prev_learned_types}')
        
        labels_all_learned_types = [idx2label[x] for x in learned_types]
        
        description_stage_loader = DataLoader(
            DescriptionDataset(args, tokenizer, labels_all_learned_types),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_description
        )
        
        dev_score = None
        no_better = 0

        if stage > 0:
            ep_time = args.task_ep_time
        else:
            ep_time = 1
            
        num_epochs = int(args.epochs * ep_time)
        
        logger.info("Start training ...")
        for ep in tqdm(range(num_epochs), desc="Epoch"):
            num_choose = [0] * model.num_experts
            if stage == 0 and args.skip_first:
                continue
            
            if args.use_mole:
                if ep < args.uniform_ep:
                    model.turn_uniform_expert(turn_on=True)
                else:
                    model.turn_uniform_expert(turn_on=False)
                    
            model.train()
            if args.gradient_checkpointing:
                # model.gradient_checkpointing_enable()
                pass
            
            wandb.log({"epoch": ep + 1 + args.epochs * stage, "stage": stage})
            
            iter_cnt = 0
            for batch in stage_loader:
                iter_cnt += 1
                optimizer.zero_grad()  
                
                train_x, train_y, train_masks, train_span, train_augment = zip(*batch)
                train_x = torch.LongTensor(train_x).to(device)
                train_masks = torch.LongTensor(train_masks).to(device)
                train_y = [torch.LongTensor(item).to(device) for item in train_y]           
                train_span = [torch.LongTensor(item).to(device) for item in train_span]
                augment_x = {}
                augment_masks = {}
                augment_y = {}
                augment_span = {}
                for aug_ids in range(args.num_augmention):
                    augment_x[aug_ids] = [torch.LongTensor(item[aug_ids][0]).to(device) for item in train_augment]
                    augment_y[aug_ids] = [torch.LongTensor(item[aug_ids][1]).to(device) for item in train_augment]
                    augment_masks[aug_ids] = [torch.LongTensor(item[aug_ids][2]).to(device) for item in train_augment]
                    augment_span[aug_ids] = [torch.LongTensor(item[aug_ids][3]).to(device) for item in train_augment]

                augment_x_list = [
                    torch.stack(value, dim=0)  # → (B, L)
                    for _, value in augment_x.items()
                ]
                augment_x_total = torch.cat(augment_x_list, dim=0).to(device)

                augment_y_total = [
                    tensor
                    for value in augment_y.values()
                    for tensor in value
                ]
                
                augment_masks_list = [
                    torch.stack(value, dim=0)  # → (B, L)
                    for _, value in augment_masks.items()
                ]
                augment_masks_total = torch.cat(augment_masks_list, dim=0).to(device)
                
                augment_span_total = [
                    tensor
                    for value in augment_span.values()
                    for tensor in value
                ]
                
                labels_for_loss_des = []
                for y in train_y:
                    for k in y:
                        if k in learned_types and k != 0:
                           labels_for_loss_des.append(idx2label[int(k)])
                           break 
                # print(labels)
                
                # if args.dataset == "ACE":
                #     return_dict = model(train_x, train_masks)
                # else: 
                return_dict = model(train_x, train_masks, train_span)
                outputs, context_feat, trig_feat = return_dict['outputs'], return_dict['context_feat'], return_dict['trig_feat']
                # DISTILL ----------------------------
                imp_mask, cur_feat_tokens_imp = return_dict['imp_mask'], return_dict['cur_feat_tokens_imp']
                # DISTILL ----------------------------
                if args.use_mole:
                    for i, num in enumerate(return_dict['num_choose']):
                        num_choose[i] += num
                # invalid_mask_op = torch.BoolTensor([item not in learned_types for item in range(args.class_num)]).to(device)
                # not from below's codes
                
                # Remove label not in event type set
                for i in range(len(train_y)):
                    invalid_mask_label = torch.BoolTensor([item not in learned_types for item in train_y[i]]).to(device)
                    train_y[i].masked_fill_(invalid_mask_label, 0)
                # outputs[:, 0] = 0
                loss, loss_ucl, loss_aug, loss_fd, loss_pd, loss_tlcl = 0, 0, 0, 0, 0, 0
                ce_y = torch.cat(train_y) # (sum of len(label), )
                ce_outputs = outputs
                if (args.ucl or args.tlcl) and (stage > 0 or (args.skip_first_cl != "ucl+tlcl" and stage == 0)):                        
                    reps = return_dict['reps']
                    bs, hdim = reps.shape
                    aug_repeat_times = args.aug_repeat_times
                    # Create dataset
                    da_x = train_x.clone().repeat((aug_repeat_times, 1))
                    da_y = train_y * aug_repeat_times
                    da_masks = train_masks.repeat((aug_repeat_times, 1))
                    da_span = train_span * aug_repeat_times
                    tk_len = torch.count_nonzero(da_masks, dim=-1) - 2
                    # Random permutation
                    perm = [torch.randperm(item).to(device) + 1 for item in tk_len]
                    
                    # Permutation
                    if args.cl_aug == "shuffle":
                        for i in range(len(tk_len)):
                            da_span[i] = torch.where(da_span[i].unsqueeze(2) == perm[i].unsqueeze(0).unsqueeze(0))[2].view(-1, 2) + 1
                            da_x[i, 1: 1+tk_len[i]] = da_x[i, perm[i]]
                    # For 25%
                    elif args.cl_aug =="RTR":
                        rand_ratio = 0.25
                        rand_num = (rand_ratio * tk_len).int()
                        
                        # Tokens that not change
                        special_ids = [103, 102, 101, 100, 0]
                        all_ids = torch.arange(model.backbone.config.vocab_size).to(device)
                        special_token_mask = torch.ones(model.backbone.config.vocab_size).to(device)
                        special_token_mask[special_ids] = 0
                        all_tokens = all_ids.index_select(0, special_token_mask.nonzero().squeeze())
                        for i in range(len(rand_num)):
                            token_idx = torch.arange(tk_len[i]).to(device) + 1
                            trig_mask = torch.ones(token_idx.shape).to(device)
                            if args.dataset == "ACE":
                                span_pos = da_span[i][da_y[i].nonzero()].view(-1).unique() - 1
                            else:
                                span_pos = da_span[i].view(-1).unique() - 1
                            # Tokens that not change
                            trig_mask[span_pos] = 0
                            token_idx_ntrig = token_idx.index_select(0, trig_mask.nonzero().squeeze())
                            replace_perm = torch.randperm(token_idx_ntrig.shape.numel())
                            replace_idx = token_idx_ntrig[replace_perm][:rand_num[i]]
                            new_tkn_idx = torch.randperm(len(all_tokens))[:rand_num[i]]
                            da_x[i, replace_idx] = all_tokens[new_tkn_idx].to(device)
                    # if args.dataset == "ACE":
                    #     da_return_dict = model(da_x, da_masks)
                    # else:
                    
                    # Hidden representaion of data augment
                    da_return_dict = model(da_x, da_masks, da_span)
                    da_outputs, da_reps, da_context_feat, da_trig_feat = da_return_dict['outputs'], da_return_dict['reps'], da_return_dict['context_feat'], da_return_dict['trig_feat']
                    # DISTILL ----------------------------
                    da_imp_mask, da_cur_feat_tokens_imp = da_return_dict['imp_mask'], da_return_dict['cur_feat_tokens_imp']
                    # DISTILL ----------------------------
                    # Contrastive loss for sentence
                    if args.ucl:
                        if not ((args.skip_first_cl == "ucl" or args.skip_first_cl == "ucl+tlcl") and stage == 0):
                            ucl_reps = torch.cat([reps, da_reps])
                            ucl_reps = normalize(ucl_reps, dim=-1)
                            Adj_mask_ucl = torch.zeros(bs * (1 + aug_repeat_times), bs * (1 + aug_repeat_times)).to(device)
                            for i in range(aug_repeat_times):
                                Adj_mask_ucl += torch.eye(bs * (1 + aug_repeat_times)).to(device)
                                Adj_mask_ucl = torch.roll(Adj_mask_ucl, bs, -1)                    
                            loss_ucl = compute_CLLoss(Adj_mask_ucl, ucl_reps, bs * (1 + aug_repeat_times), args, device)
                            
                    # Contrastive loss for trigger
                    if args.tlcl:
                        if not ((args.skip_first_cl == "tlcl" or args.skip_first_cl == "ucl+tlcl") and stage == 0):
                            tlcl_feature = torch.cat([trig_feat, da_trig_feat])
                            # tlcl_feature = trig_feat
                            tlcl_feature = normalize(tlcl_feature, dim=-1)
                            tlcl_lbs = torch.cat(train_y + da_y)
                            # tlcl_lbs = torch.cat(train_y)
                            mat_size = tlcl_feature.shape[0]
                            tlcl_lbs_oh = F.one_hot(tlcl_lbs).float()
                            # tlcl_lbs_oh[:, 0] = 0 # whether to compute negative distance
                            Adj_mask_tlcl = torch.matmul(tlcl_lbs_oh, tlcl_lbs_oh.T)
                            Adj_mask_tlcl = Adj_mask_tlcl * (torch.ones(mat_size) - torch.eye(mat_size)).to(device)
                            loss_tlcl = compute_CLLoss(Adj_mask_tlcl, tlcl_feature, mat_size, args, device)
                    loss = loss + loss_ucl + loss_tlcl*args.weight_loss_tlcl
                    if args.joint_da_loss == "ce" or args.joint_da_loss == "mul":
                        ce_y = torch.cat(train_y + da_y)
                        ce_outputs = torch.cat([outputs, da_outputs])


                
                    # outputs[i].masked_fill_(invalid_mask_op, torch.Tensor([float("-inf")]).squeeze(0))
                # if args.dataset == "ACE":
                loss_des_cl = torch.tensor(0.0, device=device)
                if args.use_description and (stage > 0 or ((not args.skip_des) and stage == 0)): 
                    
                    reps = trig_feat
                    descriptions_representations = {}
                    final_description_res = {}
                    
                    model.eval()
                    with torch.no_grad():
                        for bt, description_batch in enumerate(description_stage_loader):

                            train_x_description, train_masks_description, keys = description_batch
                            train_x_description = torch.LongTensor(train_x_description).to(device)
                            train_masks_description = torch.LongTensor(train_masks_description).to(device)
                
                            return_dict_description = model.forward_cls(train_x_description, train_masks_description)
                            context_feat_descriptions = return_dict_description
                            for key, context_feat_description in zip(keys, context_feat_descriptions):
                                if key not in descriptions_representations:
                                    descriptions_representations[key] = []
                                descriptions_representations[key].append(context_feat_description)
                                                                
                        for key, value in descriptions_representations.items():
                            feature = torch.stack(value, dim=0)
                            temp = torch.mean(feature, dim=0)
                            final_description_res[key] = temp
                            
                    model.train()
                    
                    if args.loss_des_type == "1":
                        negative_dict = find_negative_labels(final_description_res)       
                        loss_des_cl = contrastive_loss_des(reps, labels_for_loss_des, final_description_res, negative_dict)       
        
                    elif args.loss_des_type == "2":
                        des_feat = []
                        des_y = []
                        for key_des, value_des in final_description_res.items():
                            des_feat.append(value_des)
                            des_y.append(key_des)
                        
                        des_feat = torch.stack(des_feat, dim=0) 
                        des_y_tensor = torch.tensor([label2idx[int(xx)] for xx in des_y],
                                                    dtype=torch.long,
                                                    device=device)
                        
                        
                        des_cl_feature = torch.cat([trig_feat, des_feat])
                        # tlcl_feature = trig_feat
                        des_cl_feature = normalize(des_cl_feature, dim=-1)
                        des_cl_lbs = torch.cat(train_y + [des_y_tensor], dim=0)
                        # tlcl_lbs = torch.cat(train_y)
                        des_mat_size = des_cl_feature.shape[0]
                        des_cl_lbs_oh = F.one_hot(des_cl_lbs).float()
                        # tlcl_lbs_oh[:, 0] = 0 # whether to compute negative distance
                        Des_adj_mask_tlcl = torch.matmul(des_cl_lbs_oh, des_cl_lbs_oh.T)
                        Des_adj_mask_tlcl = Des_adj_mask_tlcl * (torch.ones(des_mat_size) - torch.eye(des_mat_size)).to(device)
                        loss_des_cl = compute_CLLoss(Des_adj_mask_tlcl, des_cl_feature, des_mat_size, args, device)
                    
                    loss = loss + loss_des_cl * args.ratio_loss_des_cl      
                    
                lgacl_loss = torch.tensor(0.0, device=device)
                if args.gpt_augmention:
                    augment_return_dict = model(augment_x_total, augment_masks_total, augment_span_total)
                    augment_trig_feat = augment_return_dict['trig_feat']
                    
                    lgacl_feature = torch.cat([trig_feat, augment_trig_feat])
                    # tlcl_feature = trig_feat
                    lgacl_feature = normalize(lgacl_feature, dim=-1)
                    lgacl_lbs = torch.cat(train_y + augment_y_total, dim=0)
                    if args.decrease_0_gpt_augmention:
                        lgacl_feature, lgacl_lbs = balance_zero_with_nonzero(lgacl_feature, lgacl_lbs, args)
                    # tlcl_lbs = torch.cat(train_y)
                    mat_size = lgacl_feature.shape[0]
                    lgacl_lbs_oh = F.one_hot(lgacl_lbs).float()
                    # tlcl_lbs_oh[:, 0] = 0 # whether to compute negative distance
                    Adj_mask_lgacl = torch.matmul(lgacl_lbs_oh, lgacl_lbs_oh.T)
                    Adj_mask_lgacl = Adj_mask_lgacl * (torch.ones(mat_size) - torch.eye(mat_size)).to(device)
                    lgacl_loss = compute_CLLoss(Adj_mask_lgacl, lgacl_feature, mat_size, args, device)
                
                loss = loss + lgacl_loss*args.ratio_loss_gpt
                
                # Loss ce of current class
                ce_outputs = ce_outputs[:, learned_types]
                if args.use_weight_ce:
                    loss_ce = CrossEntropyLossWithWeight(ce_outputs, ce_y, alpha=args.alpha_ce)
                else:
                    loss_ce = criterion_ce(ce_outputs, ce_y)
                loss = loss + loss_ce
                w = len(prev_learned_types) / len(learned_types)

                # Loss ce of old class

                if args.rep_aug != "none" and stage > 0:
                    outputs_aug, aug_y = [], []
                    for e_batch in e_loader:
                        exemplar_x, exemplars_y, exemplar_masks, exemplar_span, exemplar_augment = zip(*e_batch)
                        exemplar_radius = [exemplars.radius[y[0]] for y in exemplars_y]
                        exemplar_x = torch.LongTensor(exemplar_x).to(device)
                        exemplar_masks = torch.LongTensor(exemplar_masks).to(device)
                        exemplars_y = [torch.LongTensor(item).to(device) for item in exemplars_y]
                        exemplar_span = [torch.LongTensor(item).to(device) for item in exemplar_span]    
                        augment_exemplars_x = {}
                        augment_exemplars_masks = {}
                        augment_exemplars_y = {}
                        augment_exemplars_span = {}
                        for aug_ids in range(args.num_augmention):
                            augment_exemplars_x[aug_ids] = [torch.LongTensor(item[aug_ids][0]).to(device) for item in exemplar_augment]
                            augment_exemplars_y[aug_ids] = [torch.LongTensor(item[aug_ids][1]).to(device) for item in exemplar_augment]
                            augment_exemplars_masks[aug_ids] = [torch.LongTensor(item[aug_ids][2]).to(device) for item in exemplar_augment]
                            augment_exemplars_span[aug_ids] = [torch.LongTensor(item[aug_ids][3]).to(device) for item in exemplar_augment]
                                    
                        if args.rep_aug == "relative":
                            aug_return_dict = model(exemplar_x, exemplar_masks, exemplar_span, torch.sqrt(torch.stack(exemplar_radius)).unsqueeze(-1))
                        else:
                            aug_return_dict = model(exemplar_x, exemplar_masks, exemplar_span, torch.sqrt(torch.stack(list(exemplars.radius.values())).mean()))
                        output_aug = aug_return_dict['outputs_aug']
                        outputs_aug.append(output_aug)
                        aug_y.extend(exemplars_y)
                    outputs_aug = torch.cat(outputs_aug)
                    if args.leave_zero:
                        outputs_aug[:, 0] = 0
                    outputs_aug = outputs_aug[:, learned_types].squeeze(-1)
                    loss_aug = criterion_ce(outputs_aug, torch.cat(aug_y))
                    # loss = loss_ce * w + loss_aug * (1 - w)
                    # loss = loss_ce * (1 - w) + loss_aug * w
                    loss = args.gamma * loss + args.theta * loss_aug
                    
                if stage > 0 and args.distill != "none":
                    prev_model.eval()
                    with torch.no_grad():
                        # DISTILL ----------------------------
                        if args.distill_imp:
                            prev_return_dict = prev_model(train_x, train_masks, train_span, None, True, imp_mask)
                        else:
                            prev_return_dict = prev_model(train_x, train_masks, train_span)
                        # DISTILL ----------------------------
                        
                        prev_outputs = prev_return_dict['outputs']
                        # DISTILL ----------------------------
                        if args.distill_imp:
                            prev_feature_add = prev_return_dict['cur_feat_tokens_imp']
                        prev_feature = prev_return_dict['context_feat']
                        # DISTILL ----------------------------

                        if args.joint_da_loss == "dist" or args.joint_da_loss == "mul":
                            outputs = torch.cat([outputs, da_outputs])
                            # DISTILL ----------------------------
                            if args.distill_imp:
                                cur_feat_tokens_imp = torch.cat([cur_feat_tokens_imp, da_cur_feat_tokens_imp])
                            # DISTILL ----------------------------
                            
                            context_feat = torch.cat([context_feat, da_context_feat])
                            
                            # DISTILL ----------------------------
                            if args.distill_imp:
                                prev_return_dict_cl = prev_model(da_x, da_masks, da_span, None, True, da_imp_mask)
                            else:
                                prev_return_dict_cl = prev_model(da_x, da_masks, da_span)
                            # DISTILL ----------------------------
                            prev_outputs_cl = prev_return_dict_cl['outputs']
                            
                            # DISTILL ----------------------------
                            if args.distill_imp:
                                prev_feature_cl_add = prev_return_dict_cl['cur_feat_tokens_imp']
                            # DISTILL ----------------------------
                            
                            prev_feature_cl = prev_return_dict_cl['context_feat']
                            
                            prev_outputs, prev_feature = torch.cat([prev_outputs, prev_outputs_cl]), torch.cat([prev_feature, prev_feature_cl])
                            # DISTILL ----------------------------
                            if args.distill_imp:
                                prev_feature_add = torch.cat([prev_feature_add, prev_feature_cl_add])
                            # DISTILL ----------------------------
                    # prev_invalid_mask_op = torch.BoolTensor([item not in prev_learned_types for item in range(args.class_num)]).to(device)
                    prev_valid_mask_op = torch.nonzero(torch.BoolTensor([item in prev_learned_types for item in range(args.class_num + 1)]).to(device))
                    if args.distill == "fd" or args.distill == "mul":
                        prev_feature = normalize(prev_feature.view(-1, prev_feature.shape[-1]), dim=-1)
                        cur_feature = normalize(context_feat.view(-1, prev_feature.shape[-1]), dim=-1)
                        loss_fd = criterion_fd(prev_feature, cur_feature, torch.ones(prev_feature.size(0)).to(device)) # TODO: Don't know whether the code is right
                        # DISTILL ----------------------------
                        if args.distill_imp:
                            prev_feature_add = normalize(prev_feature_add.view(-1, prev_feature.shape[-1]), dim=-1)
                            cur_feature_add = normalize(cur_feat_tokens_imp.view(-1, prev_feature.shape[-1]), dim=-1)
                            loss_fd += criterion_fd(prev_feature_add, cur_feature_add, torch.ones(prev_feature_add.size(0)).to(device)) * args.ratio_loss_distill
                        # DISTILL ----------------------------
                    else:
                        loss_fd = 0
                    if args.distill == "pd" or args.distill == "mul":
                        T = args.temperature
                        if args.leave_zero:
                            prev_outputs[:, 0] = 0
                        prev_outputs = prev_outputs[:, prev_valid_mask_op].squeeze(-1)
                        cur_outputs = outputs[:, prev_valid_mask_op].squeeze(-1)
                        # prev_outputs[i].masked_fill_(prev_invalid_mask_op, torch.Tensor([float("-inf")]).squeeze(0))
                        prev_p = torch.softmax(prev_outputs / T, dim= -1)
                        p = torch.log_softmax(cur_outputs / T, dim = -1)
                        loss_pd = -torch.mean(torch.sum(prev_p * p, dim = -1), dim = 0)
                    else:
                        loss_pd = 0
                    # loss_pd = criterion_pd(torch.cat([item / T for item in outputs]), torch.cat([item / T for item in prev_outputs]))
                    if args.dweight_loss and stage > 0:
                        loss = loss * (1 - w) + (loss_fd + loss_pd) * w
                    else:
                        loss = loss + args.alpha * loss_fd + args.beta * loss_pd
                
                if args.use_mole and not model.uniform_expert:
                    loss = loss + args.entropy_weight * return_dict['entropy_loss'] + args.load_balance_weight * return_dict['load_balance_loss']
                model.unfreeze_lora()
                loss.backward()
                total_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                if args.print_trainable_params:
                    model.print_trainable_parameters()
                
                optimizer.step() 

                wandb.log({
                            "loss_ce_task": loss_ce,
                            "entropy_loss": return_dict.get('entropy_loss', 0),
                            "load_balance_loss": return_dict.get('load_balance_loss', 0),
                            "loss_des_cl": loss_des_cl,
                            "loss_aug": loss_aug,
                            "loss_fd": loss_fd,
                            "loss_pd": loss_pd,
                            "loss_all": loss,
                            "total_norm": total_norm,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                        })
            scheduler.step()
            logger.info(f"Num choose: {num_choose}")
            if ((ep + 1) % int(args.eval_freq*ep_time) == 0 and args.early_stop and ((ep + 1) >= args.skip_eval_ep*ep_time or stage > 0)) or (ep + 1) == num_epochs: # TODO TODO
                # Evaluation process
                logger.info("Evaluation process ...")
                model.eval()
                with torch.no_grad():
                    if args.single_label:
                        eval_dataset = collect_eval_sldataset(args.dataset, args.data_root, 'test', label2idx, None, [i for item in streams for i in item], args)
                    else:
                        eval_dataset = collect_dataset(args.dataset, args.data_root, 'test', label2idx, None, [i for item in streams for i in item], args)
                    eval_loader = DataLoader(
                        dataset=eval_dataset,
                        shuffle=True,
                        batch_size=args.eval_batch_size,
                        collate_fn=lambda x:x)
                    calcs = Calculator()
                    num_choose = [0] * model.num_experts
                    for batch in tqdm(eval_loader, desc="Eval"):
                        eval_x, eval_y, eval_masks, eval_span = zip(*batch)
                        eval_x = torch.LongTensor(eval_x).to(device)
                        eval_masks = torch.LongTensor(eval_masks).to(device)
                        eval_y = [torch.LongTensor(item).to(device) for item in eval_y]
                        eval_span = [torch.LongTensor(item).to(device) for item in eval_span]  
                        eval_return_dict = model(eval_x, eval_masks, eval_span, train=False)
                        if args.use_mole:
                            for i, num in enumerate(eval_return_dict['num_choose']):
                                num_choose[i] += num
                        eval_outputs = eval_return_dict['outputs']
                        valid_mask_eval_op = torch.BoolTensor([idx in learned_types for idx in range(args.class_num + 1)]).to(device)
                        for i in range(len(eval_y)):
                            invalid_mask_eval_label = torch.BoolTensor([item not in learned_types for item in eval_y[i]]).to(device)
                            eval_y[i].masked_fill_(invalid_mask_eval_label, 0)
                        if args.leave_zero:
                            eval_outputs[:, 0] = 0
                        eval_outputs = eval_outputs[:, valid_mask_eval_op].squeeze(-1)
                        calcs.extend(eval_outputs.argmax(-1), torch.cat(eval_y))
                        
                    bc, (precision, recall, micro_F1) = calcs.by_class(learned_types)
                    wandb.log({
                        f"precision": precision,
                        f"recall": recall,
                        f"micro_F1": micro_F1,
                    })
                    
                    
                    logger.info(f'marco F1 {micro_F1}')
                    logger.info(f"bc:{bc}")
                    logger.info(f"Num choose: {num_choose}")
                    
                    if ep + 1 == num_epochs:
                        logger.info("Final model with dev_score: " + str(micro_F1))
                        dev_scores_ls.append(micro_F1)
                        logger.info(f"Dev scores list: {dev_scores_ls}")
                        
                        
        for tp in streams_indexed[stage]:
            if not tp == 0:
                labels.pop(labels.index(tp))
                
        if args.save_dir and local_rank == 0:
            save_stage = stage + 1
            state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'stage':save_stage, 
                            'labels':labels, 'learned_types':learned_types, 'prev_learned_types':prev_learned_types}
            save_pth = os.path.join(args.save_dir, "perm" + str(args.perm_id))
            cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            save_name = f"stage_{save_stage}_{cur_time}.pth"
            if not os.path.exists(save_pth):
                os.makedirs(save_pth)
            logger.info(f'state_dict saved to: {os.path.join(save_pth, save_name)}')
            torch.save(state, os.path.join(save_pth, save_name))
            os.remove(e_pth)
            logger.info(f"Best model saved to {save_pth}")
            
    
    wandb.finish()