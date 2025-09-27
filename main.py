from classifier.train import train
from configs import parse_arguments
import wandb
import os
import torch.multiprocessing as mp

if __name__ == "__main__":
    args = parse_arguments()
    
    wandb.login(key='91f6b895fd6f5fcbf66ed4ca741eb048dd6de262')
        
    if args.parallel == 'DDP':
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        mp.spawn(train,
            args=(args, ),
            nprocs=args.world_size,
            join=True)
    else:
        train(0, args)