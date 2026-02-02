import numpy as np

class CurriculumManager:
    """
    Manages the difficulty level of the training data.
    Progresses from small, sparse graphs to large, dense ones.
    """
    def __init__(self, min_vars=20, max_vars=50, max_levels=30, stability_patience=5):
        self.min_vars = min_vars
        self.max_vars = max_vars
        self.max_levels = max_levels
        self.stability_patience = stability_patience
        
        self.current_level = 0
        self.stability_counter = 0
        self.best_metric = float('inf')
        
    def get_current_params(self):
        """
        Returns parameters for the current level.
        """
        progress = self.current_level / (self.max_levels - 1)
        
        # Linear interpolation for variables
        n_vars = int(self.min_vars + (self.max_vars - self.min_vars) * progress)
        
        # Density scaling (Harder = denser)
        density_min = 0.15 + (0.10 * progress)
        density_max = 0.25 + (0.10 * progress)
        
        # Intervention Range (Harder = wider values)
        int_range = 2.0 + (8.0 * progress)
        
        # Complexity scaling
        complexity = 0.1 + (0.9 * progress)
        
        # Mechanism Set Progression (DSR-PFN Addition)
        # Level 0-10: simple (linear, quadratic)
        # Level 10-20: medium (+ sigmoid, abs, sqrt)
        # Level 20-30: complex (all mechanisms)
        if self.current_level < 10:
            mechanism_set = 'simple'
        elif self.current_level < 20:
            mechanism_set = 'medium'
        else:
            mechanism_set = 'complex'
        
        return {
            "max_vars": n_vars,
            "density_min": density_min,
            "density_max": density_max,
            "intervention_range": int_range,
            "complexity": complexity,
            "mechanism_set": mechanism_set
        }

    def get_benchmark_params(self):
        """
        Returns a dictionary of fixed difficulty configs for cross-validation.
        """
        benchmarks = {}
        
        # Easy: Minimum vars, low density
        benchmarks['easy'] = {
            "max_vars": self.min_vars,
            "density_min": 0.15,
            "density_max": 0.20,
            "intervention_range": 2.0,
            "complexity": 0.0,
            "mechanism_set": 'simple'
        }
        
        # Medium: Mid-point vars
        mid_vars = int((self.min_vars + self.max_vars) / 2)
        benchmarks['medium'] = {
            "max_vars": mid_vars,
            "density_min": 0.20,
            "density_max": 0.30,
            "intervention_range": 5.0,
            "complexity": 0.5,
            "mechanism_set": 'medium'
        }
        
        # Hard: Max vars, high density
        benchmarks['hard'] = {
            "max_vars": self.max_vars,
            "density_min": 0.30,
            "density_max": 0.40,
            "intervention_range": 10.0,
            "complexity": 1.0,
            "mechanism_set": 'complex'
        }
        
        return benchmarks

    def update(self, val_mae, val_f1, benchmark_maes=None):
        """
        Check if we should level up.
        benchmark_maes: List of MAE scores from cross-difficulty benchmarks.
        Returns: (leveled_up: bool, reset_lr: bool)
        """
        # Simple gating logic based on MAE (since our main goal is Delta Prediction)
        # Relax threshold as difficulty increases
        params = self.get_current_params()
        n_vars = params['max_vars']
        
        # Tightened Thresholds for Phase 3 (Normalized Data)
        # Random guessing on N(0,1) gives MAE ~0.8.
        # We want to ensure the model is actually predicting deltas well.
        # Theoretical min is 0.0. Good perf is < 0.2.
        # We set loose gates to ensure progress but meaningful learning.
        if n_vars <= 25: thresh = 0.45   # Was 8.0 (Normalized 0.45 implies R2 > 0.5)
        elif n_vars <= 35: thresh = 0.55 # Was 18.0
        else: thresh = 0.65             # Was 30.0
        
        if val_mae < thresh and val_f1 > 0.4:
            # Check benchmarks if provided (Robustness Check)
            if benchmark_maes:
                avg_bench = np.mean(benchmark_maes)
                # Logic: Don't level up if we are completely failing harder tasks (e.g. MAE > 1.5*thresh)
                # This prevents "Local Minima" on easy tasks.
                if avg_bench > (thresh * 1.5):
                    self.stability_counter = 0 
                    return False, False
            
            self.stability_counter += 1
        else:
            self.stability_counter = 0
            
        if self.stability_counter >= self.stability_patience and self.current_level < self.max_levels - 1:
            self.current_level += 1
            self.stability_counter = 0
            return True, True # Level Up, Reset LR
            
        return False, False

    def state_dict(self):
        return {
            "current_level": self.current_level,
            "stability_counter": self.stability_counter,
            "best_metric": self.best_metric
        }

    def load_state_dict(self, state_dict):
        self.current_level = state_dict.get("current_level", 0)
        self.stability_counter = state_dict.get("stability_counter", 0)
        self.best_metric = state_dict.get("best_metric", float('inf'))
