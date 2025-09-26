import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

class Exemplars():
    def __init__(self, args) -> None:
        # self.exemplars = {}
        self.learned_nums = 0
        self.memory_size = args.enum * self.learned_nums if args.fixed_enum else args.enum
        self.exemplars_x = []
        self.exemplars_mask = []
        self.exemplars_y = []
        self.exemplars_span = []
        self.exemplars_augment = []
        self.radius = {}
        self.args = args

    def __len__(self):
        return self.memory_size
    
    def get_exemplar_loader(self):
        x = [item for t in self.exemplars_x for item in t]
        y = [item for t in self.exemplars_y for item in t]
        mask = [item for t in self.exemplars_mask for item in t]
        span = [item for t in self.exemplars_span for item in t]  
        aug = [item for t in self.exemplars_augment for item in t]      
        return (x, mask, y, span, aug ,self.radius)

    def rm_exemplars(self, exemplar_num):
        if self.exemplars_x != [] and exemplar_num > len(self.exemplars_x[0]):
            self.exemplars_x = [i[:exemplar_num] for i in self.exemplars_x]
            self.exemplars_mask = [i[:exemplar_num] for i in self.exemplars_mask]
            self.exemplars_y = [i[:exemplar_num] for i in self.exemplars_y]
            self.exemplars_span = [i[:exemplar_num] for i in self.exemplars_span]
            self.exemplars_augment = [i[:exemplar_num] for i in self.exemplars_augment]
            
    def set_exemplars(self, model: nn.Module, exemplar_loader: DataLoader, learned_nums, device):
        self.learned_nums = learned_nums - 1 if learned_nums > 0 else 1
        if self.args.fixed_enum:
            exemplar_num = self.args.enum
            self.memory_size = exemplar_num * self.learned_nums
        else:
            exemplar_num = int(self.memory_size / self.learned_nums)
            self.rm_exemplars(exemplar_num)
        rep_dict, data_dict = {}, {}
        model.eval()
        with torch.no_grad():
            print("Setting exemplars, loading exemplar batch:")
            for batch in tqdm(exemplar_loader):
                data_x, data_y, data_masks, data_span, data_aug = zip(*batch)
                # tensor_x = torch.LongTensor(data_x).to('cpu')
                # tensor_masks = torch.LongTensor(data_masks).to('cpu')
                tensor_x = torch.LongTensor(data_x).to(device)
                tensor_masks = torch.LongTensor(data_masks).to(device)
                if self.args.parallel == 'DP':
                    rep = model.module.forward_backbone(tensor_x, tensor_masks)
                else:
                    rep = model.forward_backbone(tensor_x, tensor_masks)

                for i in range(rep.size(0)):
                    for j, label in enumerate(data_y[i]):
                        if label != 0:
                            if not label in rep_dict:
                                rep_dict[label], data_dict[label] = [], []
                            # data_dict[label].append([data_x[i], data_y[i], data_masks[i], data_span[i]])
                            data_dict[label].append([data_x[i], [label], data_masks[i], [data_span[i][j]], data_aug[i]])
                            rep_dict[label].append(rep[i, 0, :].squeeze(0))
                # if len(rep_dict) > 20: # TODO: test use
                #     break
            for l, reps in rep_dict.items():
                reps = torch.stack(reps)
                radius = torch.mean(torch.var(reps, dim=0)) if reps.shape[0] > 1 else torch.tensor(0).to(device)
                # dt, lb, sp = zip(*data_dict[l])
                data_ls = data_dict[l]
                if exemplar_num > reps.size(0): # if reps num is not enough, up sampling 
                    repeat_times = int(exemplar_num / reps.size(0)) + 1
                    reps = reps.repeat(repeat_times, 1)
                    data_ls = data_ls * repeat_times

                # data_ls = np.asarray(data_ls)
                data_ls = list(data_ls)
                                
                prototype_rep = reps.mean(0)
                dist = torch.sqrt(torch.sum(torch.square(prototype_rep - reps), dim=1))
                reps_num = exemplar_num

                topk_dist_idx = torch.topk(dist, reps_num, largest=False).indices.to('cpu')
                # self.exemplars[label] = torch.cat([self.exemplars[label], reps[topk_dist_idx, :]], 0)
                # data_topk = dt[topk_dist_idx]
                # label_topk = lb[topk_dist_idx]
                # span_topk = sp[topk_dist_idx]
                
                
                # data_topk = data_ls[list(topk_dist_idx)]
                data_topk = [data_ls[idx] for idx in topk_dist_idx]
                
                self.exemplars_x.append([item[0] for item in data_topk])
                self.exemplars_y.append([item[1] for item in data_topk])
                self.exemplars_mask.append([item[2] for item in data_topk])
                self.exemplars_span.append([item[3] for item in data_topk])
                self.exemplars_augment.append([item[4] for item in data_topk])
                
                self.radius[l] = radius
        
    def build_stage_loader(self, dataset, collate_fn=lambda x:x):
        x = [item for t in self.exemplars_x for item in t]
        y = [item for t in self.exemplars_y for item in t]
        mask = [item for t in self.exemplars_mask for item in t]
        span = [item for t in self.exemplars_span for item in t]    
        aug = [item for t in self.exemplars_augment for item in t]  
        dataset.extend(x, y, mask, span, aug)
        return DataLoader(dataset=dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)