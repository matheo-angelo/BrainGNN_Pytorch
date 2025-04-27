import torch
import numpy as np
from torch.distributions import Categorical



def instance_sparsity(m, M):
    
    # M is the number of nodes in the graph
    # m is the number of nodes in the mask
    
    return 1 - m/M

def instance_fidelity_plus(model, data, mask, outputs):
    
    # Calculates Fidelity+_prob, as exemplified in the GNN XAI taxonomic survey paper
    
    model_prob = outputs[0].max(1)[0]
    data.x[mask] = 0
    outputs= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
    masked_model_prob = outputs[0].max(1)[0]
    return torch.abs(model_prob - masked_model_prob)	

def instance_fidelity_minus(model, data, mask, outputs):
    
    # Calculates Fidelity-_prob, as exemplified in the GNN XAI taxonomic survey paper
    
    model_prob = outputs[0].max(1)[0]
    old_data_x = data.x.clone()
    data.x.zero_()
    data.x[mask] = old_data_x[mask]
    
    outputs= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
    masked_model_prob = outputs[0].max(1)[0]
    return 1 - torch.abs(model_prob - masked_model_prob)	

def explain_model(model, explain_loader, sample_indices, n_roi, biomarker_size=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    explanation_list = []
    sparsity_sum = 0
    fidelity_plus_sum = 0
    fidelity_minus_sum = 0
    mask_size = None
    autism_mask_distribution = torch.zeros(n_roi, device=device)

    for i, data in enumerate(explain_loader):

        model.explain()
        data = data.to(device)
        outputs= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)

        pred = outputs[0].max(1)[1] # 0 is autism, 1 is control

        node_mask = outputs[-1]
        node_mask, _ = torch.sort(node_mask)
        sample_idx = sample_indices[i]

        explanation_list.append((sample_idx, node_mask.tolist()))
        mask_size = len(node_mask)

        sparsity = instance_sparsity(mask_size, n_roi)
        sparsity_sum += sparsity
        
        if pred==0: 
            autism_mask_distribution[node_mask] += 1

        # Calculate fidelity
        fidelity_plus = instance_fidelity_plus(model, data, node_mask, outputs)
        fidelity_plus_sum += fidelity_plus

        fidelity_minus = instance_fidelity_minus(model, data, node_mask, outputs)
        fidelity_minus_sum += fidelity_minus

    sparsity = sparsity_sum	/ len(explain_loader)
    
    fidelity_plus = (fidelity_plus_sum	/ len(explain_loader)).item()
    
    fidelity_minus = (fidelity_minus_sum	/ len(explain_loader)).item()
    
    biomarker = (torch.topk(autism_mask_distribution, biomarker_size, sorted=True)[1]).tolist()
    biomarker_consistency = autism_mask_distribution[biomarker].sum() / autism_mask_distribution.sum()

    return explanation_list, fidelity_plus, fidelity_minus, sparsity, biomarker_consistency, biomarker
