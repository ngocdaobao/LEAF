import json, os
import torch
import torch.nn.functional as F
def balance_zero_with_nonzero(
    tlcl_feature: torch.Tensor,
    tlcl_lbs: torch.Tensor,
    args
) -> tuple[torch.Tensor, torch.Tensor]:

    zero_idx = (tlcl_lbs == 0).nonzero(as_tuple=False).flatten()
    scale_factor = args.class_num // args.task_num
    keep_zero_count = len(zero_idx) // scale_factor
    num_non_zero = tlcl_lbs.size(0) - len(zero_idx)

    if len(zero_idx) <= keep_zero_count:
        filtered_feature = tlcl_feature
        filtered_lbs = tlcl_lbs
    else:
        perm = torch.randperm(len(zero_idx), device=tlcl_lbs.device)
        drop_idx = zero_idx[perm[:len(zero_idx) - keep_zero_count]]

        mask = torch.ones(tlcl_lbs.size(0), dtype=torch.bool, device=tlcl_lbs.device)
        mask[drop_idx] = False

        filtered_feature = tlcl_feature[mask]
        filtered_lbs = tlcl_lbs[mask]

    # Flatten feature: [N, D] → [N*D]
    # flattened_feature = filtered_feature.view(-1)
    return filtered_feature, filtered_lbs

def collate_description(batch):
    tokens, masks, keys = zip(*batch)
    return list(tokens), list(masks), list(keys)

def contrastive_loss_des(reps, targets, descriptions, negative_dict, temperature=5):
    device = reps.device
        
    desc_list = torch.stack([descriptions[int(label)] for label in targets]).to(device)  # (N, D)
    
    idx2idmatrix = {}
    all_descriptions = []
    for idx, (id_rel, embed) in enumerate(descriptions.items()):
        all_descriptions.append(embed)
        idx2idmatrix[id_rel] = idx
    all_descriptions = torch.stack(all_descriptions, dim=0).to(device)
    
    similarities = sim(reps, all_descriptions) / temperature  # (N, M)
    
    pos_sim = sim(reps, desc_list).diag()  # (N,)
    
    expanded_negs = []
    for label in targets:
        neg_indices = []
        for neg_label in negative_dict[int(label)]:
            neg_indices.append(idx2idmatrix[neg_label])
        expanded_negs.append(neg_indices)

    negs = torch.tensor(expanded_negs, device=device)
    
    neg_sims = similarities[torch.arange(len(targets)).unsqueeze(1), negs] # (N, num_negs)
    
    loss = -torch.log(torch.sigmoid(pos_sim.unsqueeze(1) - neg_sims).mean(dim=1)).mean()
    
    return loss.mean()

def compute_CLLoss(Adj_mask, reprs, matsize, args, device): # compute InfoNCELoss
    logits_cl = torch.div(torch.matmul(reprs, reprs.T), args.cl_temp)
    if args.sub_max:
        logits_max_cl, _ = torch.max(logits_cl, dim=-1, keepdim=True)
        logits_cl = logits_cl - logits_max_cl
    exp_logits_cl = torch.exp(logits_cl)
    denom_cl = torch.sum(exp_logits_cl * (1 - torch.eye(matsize).to(device)), dim = -1) 
    log_prob_cl = -torch.mean((logits_cl - torch.log(denom_cl)) * Adj_mask, dim=-1)
    return torch.mean(log_prob_cl[log_prob_cl > 0])

def collect_from_json(dataset, root, split, args):
    key = None
    default = ['train', 'dev', 'test']
    if split == "train":
        pth = os.path.join(root, dataset, "perm"+str(args.perm_id), f"{dataset}_{args.task_num}task_{args.class_num // args.task_num}way_{args.shot_num}shot.{split}.jsonl")
    elif split in ['dev', 'test']:
        pth = os.path.join(root, dataset, f"{dataset}.{split}.jsonl")
    elif split == "stream":
        pth = os.path.join(root, dataset, f"stream_label_{args.task_num}task_{args.class_num // args.task_num}way.json")
    else:
        raise ValueError(f"Split \"{split}\" value wrong!")
    if not os.path.exists(pth):
        raise FileNotFoundError(f"Path {pth} do not exist!")
    else:
        print(f"Opening path: {pth}")
        with open(pth) as f:
            if pth.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
                if split == "train":
                    key = [list(i.keys()) for i in data]
                    data = [list(i.values()) for i in data]
                    
            else:
                data = json.load(f)
    return data, key

def sim(x, y):
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    
    return torch.mm(x, y.t())

@torch.no_grad()
def find_negative_labels(description_res, k=2):
    negative_dict = dict()
    description_matrix = []
    
    rel2id = dict()
    with torch.no_grad():
        for idx, (key, description) in enumerate(description_res.items()):
            rel2id[idx] = key
            description_matrix.append(description)
        
        
    description_matrix = torch.stack(description_matrix, dim=0)

    similarities = sim(description_matrix, description_matrix) / 5  # (N, M)

    _, topk_indices = torch.topk(similarities, k=min(k+1,description_matrix.shape[0]), dim=1)

    topk_indices = topk_indices[:, 1:].tolist()
    
    for i in range(len(topk_indices)):
        new_topk_indices = [rel2id[j] for j in topk_indices[i]]
        negative_dict[rel2id[i]] = new_topk_indices
    return negative_dict
