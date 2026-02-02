import argparse
import csv
from src.analysis.dag_plots import plot_dag_comparison
import os
import random

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.data.CausalDataset import CausalDataset
from src.data.SCMGenerator import SCMGenerator
from src.data.collate import collate_context_batch
from src.models.CausalTransformer import CausalTransformer
from src.training.curriculum import CurriculumManager
from src.training.loss import causal_loss_fn
from src.training.metrics import compute_f1, compute_mae, compute_shd, compute_tpr_fdr

try:
    from rich import print as rprint
    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
    from rich.table import Table

    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False


def setup_ddp():
    """Initialize torch.distributed when launched under torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            # Respect CUDA_VISIBLE_DEVICES masking per process
            n_devices = torch.cuda.device_count()
            if n_devices > 0:
                torch.cuda.set_device(local_rank % n_devices)

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        return local_rank

    return 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_validation_set(num_vars, device, edge_prob=0.2, intervention_prob=0.5, intervention_scale=1.0, complexity=0.2, mechanism_set='simple'):
    """
    Generates a fixed validation set for a specific number of variables.
    Returns a DataLoader.
    """
    # Fixed parameters for validation to be a fair benchmark
    gen = SCMGenerator(
        num_nodes=num_vars, 
        edge_prob=edge_prob, 
        noise_scale=1.0, # Fixed noise for comparability
        intervention_prob=intervention_prob,
        complexity=complexity,  # Pass complexity to SCMGenerator
        mechanism_set=mechanism_set  # Pass mechanism_set for progressive complexity
    )
    # Update Generator's intervention values scale
    # Default is usually (-2, 2) roughly. If scale=5.0, we want (-10, 10).
    # SCMGenerator.intervention_values logic might need checking, but usually it's fixed.
    # Actually SCMGenerator line 186 in generator code: loop_values = [v * intervention_scale for v in self.intervention_values]
    # So we pass it to `generate_pipeline`? 
    # `CausalDataset` calls `generate_pipeline`.
    # `CausalDataset` accepts `intervention_scale_range`. 
    # If we want FIXED scale for validation, we should set range to (scale, scale).
    
    # Generate 16 fixed graphs for validation (Speed Optimization)
    dataset = CausalDataset(
        gen, 
        num_nodes_range=(num_vars, num_vars),
        samples_per_graph=32, # Reduced from 64
        infinite=False, 
        validation_graphs=4, # Reduced from 16 for Speed (~100 batches total)
        intervention_prob=intervention_prob,
        intervention_scale_range=(intervention_scale, intervention_scale) # Fixed scale
    )
    return DataLoader(dataset, batch_size=4, collate_fn=collate_context_batch)

def evaluate_loader(model, loader, device, description="Validating"):
    """
    Evaluates the model on a given dataloader.
    Returns a dictionary of aggregated metrics.
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_nll = 0.0
    all_logits = []
    all_adj = []
    num_batches = 0
    
    with torch.no_grad():
        # Setup Progress Bar
        progress_ctx = None
        if RICH_AVAILABLE and description:
            from rich.progress import SpinnerColumn
            progress_ctx = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                transient=True
            )
            progress_ctx.start()
            task_id = progress_ctx.add_task(description, total=None) 
            
        try:
            for i, batch in enumerate(loader):
                if progress_ctx: progress_ctx.update(task_id, advance=1, description=f"{description} (Batch {i+1})")
                
                base = batch['base_samples'].to(device)
                int_s = batch['int_samples'].to(device)
                target = batch['target_row'].to(device)
                mask = batch['int_mask'].to(device)
                idx = batch['int_node_idx'].to(device)
                
                # Forward
                deltas, logits = model(base, int_s, target, mask)
                
                mse = (deltas - batch['delta'].to(device))**2

                # Compute NLL (Gaussian Assumption with Unit Variance)
                nll_val = 0.5 * (mse.mean().item() + 1.837877)
                total_nll += nll_val

                total_loss += mse.mean().item() 
                
                # Compute MAE
                total_mae += compute_mae(deltas, batch['delta'].to(device))
                
                # Compute MAE
                total_mae += compute_mae(deltas, batch['delta'].to(device))
                
                # Collect Structure Predictions (CPU to save GPU memory)
                all_logits.append(logits.cpu())
                all_adj.append(batch['adj'].cpu())
                num_batches += 1
                
        finally:
            if progress_ctx: progress_ctx.stop()
            
    # Aggregate results
    if num_batches == 0:
        return {'mae': 0.0, 'f1': 0.0, 'shd': 0.0, 'nll': 0.0, 'tpr': 0.0, 'fdr': 0.0, 'loss': 0.0}
    
    avg_nll = total_nll / num_batches

    # Concatenate all logits/adj
    all_logits = torch.cat(all_logits, dim=0)
    all_adj = torch.cat(all_adj, dim=0)
    
    avg_mae = total_mae / num_batches
    
    # 1. Standard Metrics (Threshold 0.0 -> Prob 0.5)
    f1_std = compute_f1(all_logits, all_adj, threshold=0.0)
    shd_std = compute_shd(all_logits, all_adj, threshold=0.0)
    tpr, fdr = compute_tpr_fdr(all_logits, all_adj, threshold=0.0)

    # 2. Optimal Threshold Search (Calibrated capability)
    from src.training.metrics import find_optimal_threshold
    best_thresh, best_f1, best_shd_opt = find_optimal_threshold(all_logits, all_adj)
    
    # Logging
    if description:
        print(f"[{description}] MAE: {avg_mae:.4f} | NLL: {avg_nll:.4f} | F1 (0.0): {f1_std:.3f} | Best F1: {best_f1:.3f} (@{best_thresh}) | SHD (Opt): {best_shd_opt:.1f}")

    return {
        'mae': avg_mae,
        'nll': avg_nll,      # Generative Metric
        'f1': best_f1,       # Report BEST F1 for curriculum
        'shd': best_shd_opt, # Report BEST SHD for curriculum
        'tpr': tpr,
        'fdr': fdr,
        'best_thresh': best_thresh,
        'f1_std': f1_std,
        'shd_std': shd_std
    }

def main():
    parser = argparse.ArgumentParser(description="ISD-CP Unified Training")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (Number of Graphs). Effective samples = batch_size * 64.")
    parser.add_argument("--lr", type=float, default=2e-4) # INCREASED 100x (Wake Up)
    parser.add_argument("--min_vars", type=int, default=20)
    parser.add_argument("--max_vars", type=int, default=50)
    parser.add_argument("--edge_prob", type=float, default=None, help="Fixed edge probability (overrides curriculum)")
    parser.add_argument("--intervention_prob", type=float, default=0.5, help="Probability of intervening on a node")
    parser.add_argument("--num_layers", type=int, default=16, help="Transformer Depth")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 step to verify pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--reuse_factor", type=int, default=1, help="Reuse each generated graph N times")
    parser.add_argument("--checkpoint_path", type=str, default="last_checkpoint.pt", help="Path to checkpoint")
    parser.add_argument("--grad_checkpoint", action="store_true", help="Enable gradient checkpointing (Saves Memory)")
    parser.add_argument("--lambda_dag", type=float, default=10.0, help="Weight for DAG structural loss")

    parser.add_argument("--lambda_h", type=float, default=0.0, help="Weight for Acyclicity loss")
    parser.add_argument("--lambda_sparse", type=float, default=0.0, help="Weight for Sparsity (L1) loss")
    parser.add_argument("--lambda_aux_moe", type=float, default=0.1, help="Auxiliary load-balancing loss weight for MoE")
    parser.add_argument("--router_tau", type=float, default=1.0, help="Gumbel-Softmax temperature for MoE router")
    parser.add_argument("--lambda_delta", type=float, default=100.0, help="Weight for Delta Prediction Loss")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient Clipping Max Norm") # Reduced to 1.0 for stability
    parser.add_argument("--loss_type", type=str, default="focal", choices=["bce", "focal"], help="DAG Loss Type")
    
    # Ablation Flags
    parser.add_argument("--ablation_no_twin_world", action="store_true", help="Disable Twin World Variance Reduction")
    parser.add_argument("--ablation_dense_moe", action="store_true", help="Use Dense MLP instead of Hard MoE")
    parser.add_argument("--ablation_no_interleaved", action="store_true", help="Use standard additive encoding")
    parser.add_argument("--ablation_no_dag", action="store_true", help="Disable DAG Head")
    parser.add_argument("--ablation_no_physics", action="store_true", help="Disable Physics Head")
    
    # DSR Architecture
    parser.add_argument("--enable_dsr", action="store_true", help="Enable Deep Structural Refinement (layer-wise graph refinement)")
    
    args = parser.parse_args()
    
    local_rank = setup_ddp()
    is_master = (local_rank == 0)
    
    device_name = "cpu"
    if torch.cuda.is_available():
        # Safety: If env restricts visible devices (e.g. 1 per proc), local_rank might exceed device_count.
        # Use modulo to map local_rank to valid device index.
        # If CUDA_VISIBLE_DEVICES is set by torchrun, device_count is usually 1, so index becomes 0.
        dev_idx = local_rank % torch.cuda.device_count()
        device_name = f"cuda:{dev_idx}"
    elif torch.backends.mps.is_available():
        device_name = "mps"
        
    device = torch.device(device_name)
    
    if is_master:
        print(f"--- ISD-CP Unified Training ---")
        print(f"Structure: Interleaved Tokens | Architecture: Hyper-Experts")
        print(f"Data: Twin World Variance Reduction")
        if args.enable_dsr:
            print(f"DSR Mode: ENABLED (Layer-wise Graph Refinement)")
        print(f"Device: {device}")
        
        # Create checkpoints directory
        os.makedirs("checkpoints", exist_ok=True)

    # Batch Size Warning
    # CausalDataset yields chunks of ~64-100 samples per item (Graph).
    # So batch_size=64 means 64 * 100 = 6400 samples!
    if args.batch_size > 4:
        print(f"\n[WARNING] Batch Size {args.batch_size} implies {args.batch_size} * GRAPHS per step.", flush=True)
        print(f"          Since each graph contains ~100 samples, this is {args.batch_size * 100} samples per batch!", flush=True)
        print(f"          If you encounter OOM, reduce --batch_size to 1, 2, or 4.\n", flush=True)

    # 1. Model
    # Max vars + buffer for embeddings
    # Increased d_model to 512 for Phase 3 "Physics-Native" Capacity
    model = CausalTransformer(
        num_nodes=args.max_vars + 5, 
        d_model=512,
        num_layers=args.num_layers,
        enable_dsr=args.enable_dsr  # DSR: Layer-wise graph refinement
    )
    model.to(device)
    
    if is_master:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters: {total_params:,}")
    
    if dist.is_initialized():
        print("Moving to DDP...")
        # find_unused_parameters=False is better if all params used.
        # Structure Search might have unused params in Engineer if mask blocks everything? No.
        # But separate heads might cause issues if one not used? 
        # Scientist always runs. Engineer always runs.
        # Checkpointing often requires find_unused_parameters=True if implemented weirdly.
        model = DDP(model, device_ids=[dev_idx], find_unused_parameters=False)
    
    # 2. Data & Curriculum
    curriculum = CurriculumManager(min_vars=args.min_vars, max_vars=args.max_vars)
    
    # 3. Training Loop (Delegate to Trainer/Curriculum loop)
    # Note: Our existing trainer.py 'train_model' is a simple loop.
    # We should refactor it slightly to accept the curriculum manager or run the loop here.
    # For V1 Unification, let's run a simple loop here.
    
    # ISSUE 1: AdamW needs weight_decay for structural sparsity
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Scheduler: Linear Warmup + Cosine Annealing (Per Step)
    # 5 Epochs Warmup * 2000 steps = 10,000 steps
    # 50 Epochs Restart * 2000 steps = 100,000 steps
    steps_per_epoch_est = 2000
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=5 * steps_per_epoch_est)
    scheduler_main = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50 * steps_per_epoch_est, T_mult=2, eta_min=1e-8
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_main], milestones=[5 * steps_per_epoch_est])
    
    start_epoch = 0
    
    # Validation Loader (Dynamic)
    val_loader = None
    val_loader = None
    current_val_vars = (-1, -1.0, -1.0) # (max_vars, edge_prob, complexity)
    
    start_epoch = 0
    
    # Resume Logic
    if args.resume and os.path.exists(args.checkpoint_path):
        if is_master:
            print(f"Resuming from {args.checkpoint_path}...")
        # Map location is important for DDP
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(args.checkpoint_path, map_location=map_location, weights_only=False)
        
        # Handle state dict for DDP (Fixing key mismatch)
        state_dict = checkpoint['model_state_dict']
        
        # Check if model is wrapped in DDP (has 'module.')
        is_ddp_model = hasattr(model, 'module')
        
        # Check if checkpoint keys have 'module.'
        ckpt_has_module = list(state_dict.keys())[0].startswith('module.')
        
        if is_ddp_model and not ckpt_has_module:
            # Add 'module.' prefix
            if is_master: print("Adapting checkpoint (Single-GPU) -> Model (DDP)...")
            new_state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            state_dict = new_state_dict
        elif not is_ddp_model and ckpt_has_module:
            # Strip 'module.' prefix
            if is_master: print("Adapting checkpoint (DDP) -> Model (Single-GPU)...")
            new_state_dict = {k[7:]: v for k, v in state_dict.items()}
            state_dict = new_state_dict
            
        # Phase 5 Transition: Checkpoint might be missing DAG Head keys.
        # We use strict=False to allow loading partial weights (Physics) while initializing DAG Head randomly.
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if is_master and len(missing_keys) > 0:
            print(f"Warning: Missing keys in checkpoint (Expected for Phase 5 Transition): {missing_keys}")
        
        # Optimizer Load: Might fail if params changed.
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                # Force verbose off to avoid warnings from resumed state
                if hasattr(scheduler, 'verbose'): scheduler.verbose = False
        except Exception as e:
            if is_master:
                print(f"Warning: Could not load optimizer state (New Parameters Added?): {e}")
                print("Starting with FRESH Optimizer for this Phase.")
        except Exception as e:
            if is_master:
                print(f"Warning: Could not load optimizer state (New Parameters Added?): {e}")
                print("Starting with FRESH Optimizer for this Phase.")
        curriculum.load_state_dict(checkpoint['curriculum_state_dict'])
        # ISSUE 19: Duplicate Curriculum Load Removed
        
        # Removed MoE Router Reset Logic

            
        # ISSUE 4: Always Initialize Router if not resuming (or even if resuming, to break collapse?)
        # User requested: "Re-initialized on Training Start"
        # If resuming, we might want to respect checkoint OR reset if collapse is suspected.
        # But for FRESH training, we MUST init.
        # The current code only does it inside `if args.resume`.
        # We move it OUTSIDE, but only if NOT resuming (or explicit reset).
        # Actually, let's duplicate the logic: if fresh start -> Init. If resume -> User said "Re-initializing... to break collapse", so we keep it there too.
            
    # Always ensure Router is initialized properly at start (Fresh)
    # Broadcast weights from Rank 0 to ensure consistency in DDP (since RNG is seeded differently per rank)
    # Removed MoE Initialization Logic


    for epoch in range(start_epoch, args.epochs):
        # Update Curriculum Stats
        params = curriculum.get_current_params()
        
        # Check if we need to regenerate validation set (Level Changed or First Run)
        # Check if we need to regenerate validation set (Level Changed or First Run)
        # ISSUE 6: Check density changes too
        curr_edge_prob = args.edge_prob if args.edge_prob is not None else params['density_max']
        
        # State key for validation cache
        val_state_key = (params['max_vars'], curr_edge_prob, params.get('complexity', 0.2))
        
        if val_state_key != current_val_vars:
            if is_master: 
                print(f"Generating new Validation Set for {params['max_vars']} vars, density {curr_edge_prob:.2f}, complexity {params.get('complexity', 0.2):.2f}...")
            
            val_loader = get_validation_set(
                params['max_vars'], 
                device, 
                edge_prob=curr_edge_prob,
                intervention_prob=args.intervention_prob, 
                intervention_scale=params['intervention_range'],
                complexity=params.get('complexity', 0.2),  # Pass complexity
                mechanism_set=params.get('mechanism_set', 'simple')  # Pass mechanism_set
            )
            current_val_vars = val_state_key
            
        if is_master:
            # Enhanced output with mechanism progression
            mech_set = params.get('mechanism_set', 'simple')
            mech_emoji = {'simple': '📊', 'medium': '📈', 'complex': '🔬'}[mech_set]
            print(f"\n{'='*70}")
            print(f"Epoch {epoch} | Level {curriculum.current_level}/30 | Vars: {params['max_vars']} | {mech_emoji} Mechanisms: {mech_set.upper()}")
            print(f"{'='*70}")
        
        # Generator with current difficulty
        gen = SCMGenerator(
            num_nodes=params['max_vars'], 
            edge_prob=args.edge_prob if args.edge_prob is not None else params['density_max'],
            noise_scale=1.0,
            intervention_prob=args.intervention_prob,
            complexity=params.get('complexity', 0.2),  # Pass complexity to SCMGenerator
            mechanism_set=params.get('mechanism_set', 'simple')  # Progressive mechanism complexity
        )
        
        dataset = CausalDataset(
            gen, 
            num_nodes_range=(params['max_vars']-1, params['max_vars']),
            samples_per_graph=64,
            reuse_factor=args.reuse_factor,
            use_twin_world=not args.ablation_no_twin_world,
            intervention_scale_range=(1.0, 10.0) # Reduced from 50.0 to 10.0 for numerical stability
        )
        
        # No DistributedSampler for IterableDataset
        # Each rank has its own process and generator state.
        # OPTIMIZATION: Multi-worker data loading to prevent GPU idle
        # num_workers=8 spawns 8 CPU threads to generate data in parallel
        # prefetch_factor=2 prefetches 2 batches ahead to avoid GPU stalls
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            collate_fn=collate_context_batch, 
            sampler=None,
            num_workers=8,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        # Train 1 Epoch (which is infinite stream, so we define steps)
        # "Infinite" Dataset: We define an epoch as 2000 steps (~64k samples)
        steps_per_epoch = 50 if args.dry_run else 2000
        
        model.train()
        total_loss = 0
        total_metrics = {
            "delta": 0.0, "dag": 0.0, "h": 0.0,
            "shd": 0.0, "f1": 0.0, "mae": 0.0,
            "tpr": 0.0, "fdr": 0.0, "aux_moe": 0.0
        }
        
        # Progress Bar (Only on Master)
        progress = None
        if is_master:
            if RICH_AVAILABLE:
                progress = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeRemainingColumn(),
                    TextColumn("[magenta]{task.fields[metrics]}")
                )
                task_id = progress.add_task(
                    f"[cyan]Epoch {epoch}", 
                    total=steps_per_epoch, 
                    metrics="Loss: ..."
                )
                progress.start()
            else:
                # ASCII Header
                print(f"Epoch {epoch} Started", flush=True)

        # Removed MoE Reset Logic

        
        i = 0 # Safety initialization
        
        for i, batch in enumerate(dataloader):
            if i >= steps_per_epoch: break
            
            # Move to device
            base = batch['base_samples'].to(device)
            int_s = batch['int_samples'].to(device)
            target = batch['target_row'].to(device)
            mask = batch['int_mask'].to(device)
            idx = batch['int_node_idx'].to(device)
            # Forward (Phase 4: Returns deltas, logits, adj, mcm_out, aux_loss)
            deltas, logits = model(base, int_s, target, mask)
            
            # Adaptive Weighting (User Request: Curriculum-linked)
            # Level 0 -> lambda_delta=100. Level 30 -> lambda_delta=1.
            # Formula: max(1.0, initial - 3.3 * level)
            current_lambda_delta = max(1.0, args.lambda_delta - (3.3 * curriculum.current_level))
            
            loss, items = causal_loss_fn(
                deltas, 
                batch['delta'].to(device), 
                logits, 
                batch['adj'].to(device),
                lambda_delta=current_lambda_delta,
                lambda_dag=args.lambda_dag,
                lambda_h=args.lambda_h,
                lambda_l1=args.lambda_sparse,
                loss_type=args.loss_type
            ) 
            
            # Removed MoE Aux Loss


            # REMOVED: Final loss safety guard that was hiding real failures
            # If loss is NaN/Inf, we MUST debug it, not mask it with a constant.
            # Masking with constant prevents gradient flow → model learns nothing.
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip) # Configurable Gradient Clipping
            optimizer.step()
            
            # ISSUE 7: Scheduler Step per Batch
            scheduler.step()
            
            # Additional Metrics
            with torch.no_grad():
                shd = compute_shd(logits, batch['adj'].to(device))
                f1 = compute_f1(logits, batch['adj'].to(device))
                mae = compute_mae(deltas, batch['delta'].to(device))
                tpr, fdr = compute_tpr_fdr(logits, batch['adj'].to(device))
            
            # Track Metrics
            total_loss += loss.item()
            for k, v in items.items():
                total_metrics[k] += v
            total_metrics['shd'] += shd
            total_metrics['f1'] += f1
            total_metrics['mae'] += mae
            total_metrics['tpr'] += tpr
            total_metrics['fdr'] += fdr
                
            if is_master:
                avg_loss = total_loss / (i + 1)
                avg_shd = total_metrics['shd'] / (i + 1)
                avg_mae = total_metrics['mae'] / (i + 1)
                avg_delta = total_metrics['delta'] / (i + 1)

                metric_str = f"L: {avg_loss:.1f} | Δ: {avg_delta:.2f} | MAE: {avg_mae:.2f} | SHD:{avg_shd:.1f}"
                
                if RICH_AVAILABLE and progress is not None:
                    progress.update(task_id, advance=1, metrics=metric_str)
                else:
                    # ASCII logic simplified
                    mech_set = params.get('mechanism_set', 'simple')
                    print(f"\rStep {i+1}/{steps_per_epoch} | {metric_str} | {mech_set[:3].upper()}", end='', flush=True)

            if args.dry_run: break
            
        # --- END OF EPOCH BLOCK (Outside Loop) ---
        if is_master:
            if progress: progress.stop()
            if not RICH_AVAILABLE: print(flush=True)
            
        # Synchronize before validation to prevent timeouts
        if dist.is_initialized():
            dist.barrier()
            
        # --- Validation Loop (Fixed Set & Benchmarks) ---
        # 1. Evaluate Current Level (All Ranks participate to sync Curriculum)
        if is_master: print(f"Validating Current Level {curriculum.current_level}...", flush=True)
        # Only show progress bar on Master
        val_desc = "Validating" if is_master else ""
        val_metrics = evaluate_loader(model, val_loader, device, description=val_desc)
        val_mae = val_metrics['mae']
        val_f1 = val_metrics['f1']
        
        if is_master:
            print(f"Val Level {curriculum.current_level} | MAE: {val_mae:.3f} | SHD: {val_metrics['shd']:.1f} | F1: {val_f1:.3f} | TPR: {val_metrics['tpr']:.2f} | FDR: {val_metrics['fdr']:.2f}")
            
            pass 

        # --- MOVED BENCHMARK LOGIC OUTSIDE is_master CHECK ---
        val_tpr = val_metrics['tpr']
        val_fdr = val_metrics['fdr']
        
        # Step Scheduler (Cosine uses epoch, not val metric)
        # Scheduler Step per Batch now
        # scheduler.step()
        
        # Calculate Epoch Metrics (Training Avg)
        i = max(1, i) # Avoid div by zero if loop didn't run
        avg_loss = total_loss / (max(1, i+1))
        # avg_mae = total_metrics['mae'] / (max(1, i+1)) # Using Validation MAE for curriculum now
        avg_shd = total_metrics['shd'] / (max(1, i+1)) 
        
        # 1. Update Curriculum (using VALIDATION Scores)
        # ISSUE 11: Sync Metrics across ranks to prevent Curriculum divergence
        with torch.no_grad():
            if dist.is_initialized():
                 # Stack metrics [mae, f1]
                 local_metrics = torch.tensor([val_mae, val_f1], device=device)
                 dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM) # Sum across ranks
                 local_metrics /= dist.get_world_size() # Average
                 val_mae = local_metrics[0].item()
                 val_f1 = local_metrics[1].item()
        
        # FIX: Run benchmarks on ALL RANKS (silently for slaves).
        
        benchmarks = curriculum.get_benchmark_params()
        benchmark_maes = []
        # if is_master: print("--- Cross-Difficulty Benchmarks ---")
        
        for level_name, b_params in benchmarks.items():
            # if is_master: print(f"  > Benchmarking {level_name.upper()} (Vars: {b_params['max_vars']})...", flush=True)
            b_loader = get_validation_set(
                b_params['max_vars'],
                device,
                edge_prob=b_params['density_max'], 
                intervention_prob=args.intervention_prob,
                intervention_scale=b_params['intervention_range'],
                mechanism_set=b_params.get('mechanism_set', 'simple')
            )
            b_metrics = evaluate_loader(model, b_loader, device, description=level_name if is_master else "")
            
            # Sync Benchmark MAE too
            with torch.no_grad():
                b_mae = torch.tensor(b_metrics['mae'], device=device)
                if dist.is_initialized():
                     dist.all_reduce(b_mae, op=dist.ReduceOp.SUM)
                b_mae /= dist.get_world_size() if dist.is_initialized() else 1
            benchmark_maes.append(b_mae.item())

            # After benchmarks, plot one sample DAG for the user (Master Only)
            if is_master:
                with torch.no_grad():
                    # Get a sample from the last benchmark loader (or any loader)
                    v_sample = next(iter(b_loader))
                    v_base = v_sample['base_samples'].to(device)
                    v_int = v_sample['int_samples'].to(device)
                    v_target = v_sample['target_row'].to(device)
                    v_mask = v_sample['int_mask'].to(device)
                    v_true_adj = v_sample['adj'].to(device)
                    
                    _, v_logits = model(v_base, v_int, v_target, v_mask)
                    plot_path = plot_dag_comparison(v_logits[0], v_true_adj[0], epoch)
                    print(f"  [Discovery] DAG Plot saved to {plot_path}", flush=True)
                
                print(f"[{level_name.upper()}] MAE: {b_mae.item():.3f} | SHD: {b_metrics['shd']:.1f} | F1: {b_metrics['f1']:.3f}")

        leveled_up, reset_lr = curriculum.update(val_mae, val_f1, benchmark_maes)
        
        # 2. Print Summary (Master Only)
        if is_master:
            # Re-fetch metrics for summary
            # Removed MoE metrics aggregation
            moe_metrics = {'entropy': 0.0, 'gini': 0.0}

            if RICH_AVAILABLE:
                table = Table(title=f"Epoch {epoch} Summary | Level {curriculum.current_level}")
                table.add_column("Metric", style="cyan", no_wrap=True)
                table.add_column("Train", style="magenta")
                table.add_column("Val (Fixed)", style="green")
                
                table.add_row("Total Loss", f"{avg_loss:.4f}", "-")
                table.add_row("MAE (L1)", f"{total_metrics['mae']/(i+1):.4f}", f"{val_mae:.4f}")
                table.add_row("SHD", f"{avg_shd:.2f}", f"{val_metrics['shd']:.2f}")
                # table.add_row("Expert Entropy", f"{moe_metrics['entropy']:.4f}", "-")
                # table.add_row("Expert Gini", f"{moe_metrics['gini']:.4f}", "-")
                table.add_row("LR", f"{optimizer.param_groups[0]['lr']:.2e}", "")
                
                rprint(table)
                if leveled_up:
                    rprint(f"[bold yellow]*** LEVEL UP! Advanced to Level {curriculum.current_level} ***[/bold yellow]")
            else:
                print(f"Ep {epoch}|L:{avg_loss:.2f}|TrMAE:{total_metrics['mae']/(i+1):.2f}|ValMAE:{val_mae:.2f}|LR:{optimizer.param_groups[0]['lr']:.2e}")
                if leveled_up:
                    print(f"*** LEVEL UP! Advanced to Level {curriculum.current_level} ***")
            
            # 2b. CSV Logging
            log_file = "training_log.csv"
            file_exists = os.path.isfile(log_file)
            with open(log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                header = ["Epoch", "Level", "LR", 
                          "Train_Loss", "Train_Delta", "Train_DAG", "Train_H", 
                          "Train_MAE", "Train_SHD", "Train_F1", "Train_TPR", "Train_FDR", 
                          "Val_MAE", "Val_SHD", "Val_F1", "Val_TPR", "Val_FDR",
                          "Expert_Entropy", "Expert_Gini"]
                if not file_exists:
                    writer.writerow(header)
                
                train_iters = max(1, i+1) # total training iterations this epoch
                writer.writerow([
                    epoch, 
                    curriculum.current_level,
                    optimizer.param_groups[0]['lr'],
                    f"{avg_loss:.4f}",
                    f"{total_metrics['delta']/train_iters:.4f}",
                    f"{total_metrics['dag']/train_iters:.4f}",
                    f"{total_metrics['h']/train_iters:.4f}",
                    f"{total_metrics['mae']/train_iters:.4f}",
                    f"{avg_shd:.4f}",
                    f"{total_metrics['f1']/train_iters:.4f}",
                    f"{total_metrics['tpr']/train_iters:.4f}",
                    f"{total_metrics['fdr']/train_iters:.4f}",
                    f"{val_mae:.4f}",
                    f"{val_metrics['shd']:.2f}",
                    f"{val_f1:.4f}",
                    f"{val_tpr:.4f}",
                    f"{val_fdr:.4f}",
                    f"{moe_metrics['entropy']:.4f}",
                    f"{moe_metrics['gini']:.4f}"
                ])
        
        # 3. Save Checkpoint (Master Only)
        if is_master:
            # 1. Save Resume Checkpoint
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'curriculum_state_dict': curriculum.state_dict(),
                'args': args
            }
            torch.save(checkpoint_data, args.checkpoint_path)
            
            # 2. Save Historical Snapshot
            snapshot_path = f"checkpoints/checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint_data, snapshot_path)
            
        if args.dry_run:
            print("Dry Run Successful.")
            break
            
    cleanup_ddp()

if __name__ == "__main__":
    main()
