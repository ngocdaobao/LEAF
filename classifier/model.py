import torch
from transformers import BertModel
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from loguru import logger
import torch.nn.functional as F


def _l2_normalize(x, dim: int = -1):
    return F.normalize(x, p=2.0, dim=dim)

def _select_important_tokens(att_last,         # (B, H, L, L)
                             att_mask,         # (B, L)
                             span=None,
                             topk_ratio: float = 0.1):
    # ---------- (1) Attention-based scores ----------------------------
    score = att_last.mean(dim=1).sum(dim=1)         # (B, L)
    score = score.masked_fill(~att_mask.bool(), -1e4)

    B, L = score.shape
    k = max(1, int(topk_ratio * L))

    imp_mask = torch.zeros_like(att_mask, dtype=torch.bool)
    topk_idx = score.topk(k, dim=1).indices         # (B, k)
    imp_mask.scatter_(1, topk_idx, True)

    # ---------- (2) Add trigger tokens (start & end only) -------------
    if span is not None:
        for b, sp in enumerate(span):
            if sp.numel() == 0:
                continue
            # sp shape (N_trig, 2): columns = [start, end]
            starts = sp[:, 0]
            ends   = sp[:, 1]
            imp_mask[b, starts] = True
            imp_mask[b, ends]   = True
    return imp_mask

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_num, num_layers=1, dropout=0.1):
        """
        Args:
            input_dim: int, kích thước đầu vào
            hidden_dim: int, kích thước lớp ẩn
            class_num: int, số lớp đầu ra
            num_layers: int, số lượng lớp ẩn (không tính lớp output)
            dropout: float, tỷ lệ dropout
        """
        super().__init__()
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, class_num))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class BertED(nn.Module):
    def __init__(self, args, backbone_path=None):
        super().__init__()
        self.is_input_mapping = args.input_map
        self.class_num = args.class_num + 1
        self.use_mole = args.use_mole
        self.use_lora = args.use_lora
        self.top_k = args.mole_top_k
        self.num_experts = args.mole_num_experts
        self.use_general_expert = args.use_general_expert
        self.uniform_expert = False
        self.general_expert_weight = args.general_expert_weight
        self.args = args
        self.topk_ratio = args.topk_ratio   # hoặc tham số hoá theo args

        # Load backbone
        if backbone_path is not None:
            self.backbone = BertModel.from_pretrained(args.backbone)
            self.input_dim = self.backbone.config.hidden_size
            self.backbone.load_state_dict(torch.load(backbone_path)) 
            logger.info(f"Load backbone from {backbone_path}")
        else:
            self.backbone = BertModel.from_pretrained(args.backbone)
            self.input_dim = self.backbone.config.hidden_size
        self.seqlen = args.max_seqlen + 2  # +2 for [CLS] and [SEP]

        # Classifier
        if args.classifier_layer > 1:
            self.hidden_dim = args.hidden_dim
            self.fc = Classifier(self.input_dim, self.hidden_dim, self.class_num,
                                 num_layers=args.classifier_layer, dropout=args.dropout)
        else:
            self.fc = nn.Linear(self.input_dim, self.class_num)

        # Optional input mapping
        if self.is_input_mapping:
            self.map_hidden_dim = 512
            self.map_input_dim = self.input_dim * 2
            self.input_map = nn.Sequential(
                nn.Linear(self.map_input_dim, self.map_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.map_hidden_dim, self.map_hidden_dim),
                nn.ReLU(),
            )
            self.fc = nn.Linear(self.map_hidden_dim, self.class_num)

        # Setup LoRA or MoLE
        if self.use_lora or self.use_mole:
            self.peft_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["query", "value"],
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.backbone = get_peft_model(self.backbone, self.peft_config, adapter_name="general_expert")

            if self.use_mole:
                for i in range(self.num_experts):
                    self.backbone.add_adapter(f"expert_{i}", self.peft_config)

                if args.gating == "softmax":
                    self.gating_layer = nn.Linear(self.input_dim, self.num_experts)
                    self.softmax = nn.Softmax(dim=-1)
                    logger.info("Gating: softmax")
                elif args.gating == "tanh":
                    self.gating_layer = nn.Sequential(
                        nn.Linear(self.input_dim, self.num_experts),
                        nn.Tanh(),
                        nn.Linear(self.num_experts, self.num_experts),
                    )
                    self.softmax = nn.Softmax(dim=-1)
                    logger.info("Gating: tanh")
                elif args.gating == "sigmoid":
                    self.gating_layer = nn.Linear(self.input_dim, self.num_experts)
                    self.softmax = nn.Sigmoid()
                    logger.info("Gating: sigmoid")

            self.backbone.print_trainable_parameters()

        print("Trainable parameters:")
        for n, p in self.named_parameters():
            if p.requires_grad:
                print(n, p.shape)
                break

    def print_trainable_parameters(self):
        print("Trainable parameters:")
        for n, p in self.named_parameters():
            if p.requires_grad:
                print(n, p.shape)
                break
            
    def unfreeze_lora(self):
        for name, param in self.backbone.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
            elif self.args.no_freeze_bert:
                param.requires_grad = True
        # logger.info("Unfreeze LoRA parameters")
        
    def turn_uniform_expert(self, turn_on=True):
        if self.uniform_expert != turn_on:
            self.uniform_expert = turn_on
            logger.info(f"Uniform expert: {turn_on}")

    def forward(self, x, masks, span=None, aug=None, train=True, imp_mask=None):
        if self.use_mole:
            return self._forward_mole(x, masks, span, aug, train, imp_mask)
        else:
            return self._forward_normal(x, masks, span, aug)

    def _forward_normal(self, x, masks, span=None, aug=None):
        out = self.backbone(x, attention_mask=masks)
        hidden = out.last_hidden_state
        return_dict = {
            'reps': hidden[:, 0, :].clone(),
            'context_feat': hidden.view(-1, hidden.shape[-1])
        }

        if span is not None:
            trig_feature = self._extract_trigger(hidden, span)
            return_dict['trig_feat'] = trig_feature
            return_dict['outputs'] = self.fc(trig_feature)

            if aug is not None:
                feature_aug = trig_feature + torch.randn_like(trig_feature) * aug
                return_dict['feature_aug'] = feature_aug
                return_dict['outputs_aug'] = self.fc(feature_aug)

        return return_dict
    
    def set_adapter(self, name):
        self.backbone.set_adapter(name)
        self.unfreeze_lora()
        
    def _forward_mole(self, x, masks, span=None, aug=None, train=True, imp_mask=None):
        B, L = x.size(0), x.size(1)
        num_heads = self.backbone.config.num_attention_heads
        return_dict = {}

        if not self.uniform_expert:
            with torch.no_grad():
                with self.backbone.disable_adapter():
                    base_output = self.backbone(x, attention_mask=masks, return_dict=True)
                    # Instance-level: use [CLS]
                    cls_embedding = base_output.last_hidden_state[:, 0, :]  # (B, H)
                    # If token-level, use base_output (B, L, H)

            # Gating
           #gating_logits = self.gating_layer(cls_embedding)  (B, E)
            # Gating for token-level selection
            gating_logits = self.gating_layer(base_output.last_hidden_state)  # (B, L, E)
            print(gating_logits.shape)

            gating_weights = self.softmax(gating_logits)
            topk_weights, topk_indices = torch.topk(gating_weights, self.top_k, dim=-1)  # (B, k), (B, k)
            


            if train:
                avg_weights = gating_weights.mean(dim=0)
                uniform = torch.full_like(avg_weights, 1.0 / self.num_experts)
                return_dict['load_balance_loss'] = torch.sum((avg_weights - uniform) ** 2)
                return_dict['entropy_loss'] = -torch.sum(gating_weights * torch.log(gating_weights + 1e-8), dim=-1).mean()
        else:
            topk_weights = torch.full((B, self.top_k), 1.0 / self.top_k).to(x.device)
            topk_indices = torch.stack([torch.randperm(self.num_experts)[:self.top_k] for _ in range(B)], dim=0).to(x.device)

        expert_outputs = [torch.zeros(B, self.seqlen, self.input_dim, device=x.device) for _ in range(self.top_k)]
        expert_outputs_attn = [torch.zeros(B, num_heads, L, L, device=x.device)
                           for _ in range(self.top_k)]
        num_choose = [0] * self.num_experts
        for k in range(self.top_k):
            expert_ids = topk_indices[:,:, k]  # (B,), (B,L,) if token-level
            weights = topk_weights[:,:, k]     # (B,), (B,L,) if token-level

        #     for expert_id in expert_ids.unique():
        #         mask = (expert_ids == expert_id)
        #         if mask.sum() == 0:
        #             continue
        #         else:
        #             num_choose[expert_id.item()] +=  mask.sum().item()

        #         self.set_adapter(f"expert_{expert_id.item()}")
        #         out = self.backbone(
        #             x[mask], attention_mask=masks[mask], output_attentions=True
        #         )                
        #         _ = out.last_hidden_state  # (N, T, H)
        #         attn_expert = out.attentions[-1]  # (N, T, H)
                

        #         weighted = weights[mask].view(-1, 1, 1) * _
        #         expert_outputs[k][mask] = weighted
                
        #         w_attn = weights[mask].view(-1, 1, 1, 1)         # (N,1,1,1)
        #         expert_outputs_attn[k][mask] = w_attn * attn_expert

        # x_out = sum(expert_outputs)
        # total_attention = sum(expert_outputs_attn)

        # Token-level MoLE
            for expert_id in range(self.num_experts):
                mask = (expert_ids == expert_id)  # (B, L) bool
                num_selected = mask.sum().item()
                if num_selected == 0:
                    continue
                num_choose[expert_id] += num_selected

                indices = torch.nonzero(mask, as_tuple=False)  # (num_selected, 2)
                batch_idx = indices[:, 0]
                token_idx = indices[:, 1]

                selected_x = x[batch_idx, token_idx, :]      # (num_selected, H)
                selected_masks = masks[batch_idx, token_idx] # (num_selected,)
                selected_weights = weights[batch_idx, token_idx] # (num_selected,)

                self.set_adapter(f"expert_{expert_id}")
                selected_x = selected_x.unsqueeze(1)      # (num_selected, 1, H)
                selected_masks = selected_masks.unsqueeze(1) # (num_selected, 1)

                out = self.backbone(
                    selected_x, attention_mask=selected_masks, output_attentions=True
                )
                _ = out.last_hidden_state  # (num_selected, 1, H)
                attn_expert = out.attentions[-1]  # (num_selected, 1, H)

                expert_outputs[k][batch_idx, token_idx, :] = selected_weights.unsqueeze(-1) * _.squeeze(1)
                expert_outputs_attn[k][batch_idx, :, token_idx, token_idx] = selected_weights.unsqueeze(-1).unsqueeze(-1) * attn_expert.squeeze(1)



        # Optional: add general expert
        if self.use_general_expert:
            self.set_adapter("general_expert")
            general_out = self.backbone(x, attention_mask=masks, output_attentions=True)
            _1 = general_out.last_hidden_state
            _2 = general_out.attentions[-1]
            x_out += self.general_expert_weight * _1
            total_attention += self.general_expert_weight * _2

        if imp_mask is None:
            with torch.no_grad():
                imp_mask = _select_important_tokens(
                    total_attention,          # (B,H,L,L)
                    masks,               # (B,L)
                    span=span,
                    topk_ratio=self.topk_ratio
                )                        # (B,L) bool


        imp_mask = imp_mask.to(x.device).bool()
        assert imp_mask.shape == masks.shape, "imp_mask size mismatch"  

        normed_out           = _l2_normalize(x_out, dim=-1)  # (B, L, D)
        cur_feat_tokens_imp  = normed_out[imp_mask]          # indexing theo mask bool

        # 3) list per-batch cho logging/debug
        # imp_tokens_list = [imp_mask[b].nonzero(as_tuple=True)[0].tolist()
                        # for b in range(B)]

        # return_dict['cur_token_imp']      = cur_token_imp.detach()
        return_dict['imp_mask']    = imp_mask
        # return_dict['cur_feat_tokens_imp'] = cur_feat_tokens_imp.detach()
        return_dict['cur_feat_tokens_imp'] = cur_feat_tokens_imp
        # ======================================================================

        return_dict['reps'] = x_out[:, 0, :].clone()
        return_dict['context_feat'] = x_out.view(-1, x_out.shape[-1])
        return_dict['num_choose'] = num_choose

        if span is not None:
            trig_feature = self._extract_trigger(x_out, span)
            return_dict['trig_feat'] = trig_feature
            return_dict['outputs'] = self.fc(trig_feature)

            if aug is not None:
                feature_aug = trig_feature + torch.randn_like(trig_feature) * aug
                return_dict['feature_aug'] = feature_aug
                return_dict['outputs_aug'] = self.fc(feature_aug)

        return return_dict

    def _extract_trigger(self, x, span):
        trig_feature = []
        for i in range(len(span)):
            if self.is_input_mapping:
                x_cdt = torch.stack([torch.index_select(x[i], 0, span[i][:, j]) for j in range(span[i].size(-1))])
                x_cdt = x_cdt.permute(1, 0, 2).contiguous().view(x_cdt.size(1), -1)
                opt = self.input_map(x_cdt)
            else:
                opt = torch.index_select(x[i], 0, span[i][:, 0]) + torch.index_select(x[i], 0, span[i][:, 1])
            trig_feature.append(opt)
        return torch.cat(trig_feature)

    def forward_backbone(self, x, masks):
        out = self.backbone(x, attention_mask=masks)
        return out.last_hidden_state

    def forward_cls(self, x, masks):
        with torch.no_grad():
            out = self.backbone(x, attention_mask=masks)
            return out.last_hidden_state[:, 0, :]

    def forward_input_map(self, x):
        return self.input_map(x)