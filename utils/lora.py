import torch
from torch import nn

class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, alpha: int, r: int = 32):
        super(LoRALayer, self).__init__()
        self.r = r
        self.alpha = alpha
        self.A = nn.Parameter(torch.zeros((r, in_features)))
        self.B = nn.Parameter(torch.zeros((out_features, r)))
        self.scaling = self.alpha / self.r
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.A, mean =0.0, std=0.01)
        nn.init.zeros_(self.B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.scaling * (x @ self.A.transpose() @ self.B)
        return result
    
class Router(nn.Moule):
    def __init__(self, input_dim: int, num_experts: int, topk: int, backbone):
        super(Router, self).__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.topk = topk
        self.router_weights = nn.Parameter(torch.randn(input_dim, num_experts))
        self.softmax = nn.Softmax(dim=-1)
        self.backbone = backbone
    
    def forward(self, x: torch.Tensor) -> list[int]:
        logits = self.backbone(x)
        scores = self.softmax(logits @ self.router_weights)
        topk_scores, topk_indices = torch.topk(scores, self.topk, dim=-1)
        topk_scores.tolist()
        topk_indices.tolist()
        topk = [(topk_indices[i], topk_scores[i]) for i in range(len(topk_indices))]
        return topk
