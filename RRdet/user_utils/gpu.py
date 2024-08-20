import torch
import math

def get_allocated_memory(device:str) -> int:
    mem = torch.cuda.max_memory_allocated(device=device)
    # mem_res = torch.cuda.memory_reserved(0)
    mem_mb = torch.tensor([int(mem) // (1024 * 1024)],
                            dtype=torch.int,
                            device=device)

    # mem_res_mb = torch.tensor([int(mem_res) // (1024 * 1024)],
    #                         dtype=torch.int,
    #                         device=device)

    return mem_mb