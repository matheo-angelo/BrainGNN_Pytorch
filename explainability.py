import torch
import numpy as np
from torch.distributions import Categorical

def instance_sparsity(m, M):
    
    # M is the maximum possible number of nodes in the mask
    # m is the number of nodes in the mask
    
    return np.log(m/M) / np.log(1/M)

def instance_fidelity(model, data, mask, outputs):
    
    # Calculates Fidelity+_prob, as exemplified in the GNN XAI taxonomic survey paper
    
    model_prob = outputs[0].max(1)[0]
    data.x[mask] = 0
    outputs= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
    masked_model_prob = outputs[0].max(1)[0]
    return model_prob - masked_model_prob	

def explain_model(model, explain_loader, sample_indices, n_roi, biomarker_size=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    explanation_list = []
    sparsity_sum = 0
    fidelity_sum = 0
    mask_size = None
    max_entropy_distribution = torch.ones(n_roi)/n_roi
    mask_distribution = torch.zeros(n_roi, device=device)

    for i, data in enumerate(explain_loader):

        model.explain()
        data = data.to(device)
        outputs= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
        node_mask = outputs[-1]
        node_mask, _ = torch.sort(node_mask)
        sample_idx = sample_indices[i]

        explanation_list.append((sample_idx, node_mask.tolist()))
        mask_size = len(node_mask)

        sparsity = instance_sparsity(mask_size, n_roi)
        sparsity_sum += sparsity
        
        mask_distribution[node_mask] += 1

          #Calculate fidelity
        fidelity = instance_fidelity(model, data, node_mask, outputs)
        fidelity_sum += fidelity

    sparsity = sparsity_sum	/ len(explain_loader)
    fidelity = (fidelity_sum	/ len(explain_loader)).item()
    mask_entropy = Categorical(probs = mask_distribution / mask_distribution.sum()).entropy()
    max_possible_entropy = Categorical(probs = max_entropy_distribution).entropy()
    normalized_mask_entropy = mask_entropy / max_possible_entropy
    biomarker = (torch.topk(mask_distribution, biomarker_size, sorted=True)[1]).tolist()

    return explanation_list, fidelity, sparsity, normalized_mask_entropy, biomarker

