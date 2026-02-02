import torch
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_dag_comparison(pred_logits, true_adj, epoch, save_dir="plots"):
    """
    Plots Predicted Probability Adjacency Matrix vs Ground Truth DAG.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Convert to Probs and take mean across batch if needed
    if pred_logits.dim() == 3:
        pred_probs = torch.sigmoid(pred_logits[0]).cpu().numpy()
        true_adj = true_adj[0].cpu().numpy()
    else:
        pred_probs = torch.sigmoid(pred_logits).cpu().numpy()
        true_adj = true_adj.cpu().numpy()
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Prediction
    im0 = axes[0].imshow(pred_probs, cmap='RdBu_r', vmin=0, vmax=1)
    axes[0].set_title(f"Predicted Discovery (Prob)")
    plt.colorbar(im0, ax=axes[0])
    
    # Plot Truth
    im1 = axes[1].imshow(true_adj, cmap='Greys', vmin=0, vmax=1)
    axes[1].set_title("Ground Truth DAG")
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/dag_comparison_epoch_{epoch}.png")
    plt.close()
    
    # Return path for display
    return f"{save_dir}/dag_comparison_epoch_{epoch}.png"
