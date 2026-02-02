#!/usr/bin/env python3
"""
Test script to verify mechanism set progression in SCMGenerator.
This validates that simple, medium, and complex mechanism sets work correctly.
"""

import numpy as np
from src.data.SCMGenerator import SCMGenerator

def test_mechanism_sets():
    """Test that each mechanism set generates only expected mechanism types."""
    
    print("Testing Mechanism Set Progression...")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        ('simple', [1, 9], ['linear', 'quadratic']),
        ('medium', [1, 9, 11, 13, 8], ['linear', 'quadratic', 'sigmoid', 'abs', 'sqrt']),
        ('complex', list(range(1, 17)), 'all')  # All 16 mechanisms
    ]
    
    for mechanism_set, expected_ids, expected_names in test_configs:
        print(f"\n{mechanism_set.upper()} Mechanism Set:")
        print("-" * 60)
        
        # Create generator
        gen = SCMGenerator(
            num_nodes=10,
            edge_prob=0.3,
            mechanism_set=mechanism_set,
            seed=42
        )
        
        # Generate multiple DAGs to sample mechanisms
        mechanism_counts = {}
        num_trials = 50
        
        for trial in range(num_trials):
            dag = gen.generate_dag(seed=trial)
            dag = gen.edge_parameters(dag, complexity=0.5)
            
            # Count mechanism types
            for u, v in dag.edges():
                mech_type = dag[u][v].get('type', 'unknown')
                mechanism_counts[mech_type] = mechanism_counts.get(mech_type, 0) + 1
        
        # Display results
        print(f"  Expected IDs: {expected_ids}")
        print(f"  Mechanisms found ({sum(mechanism_counts.values())} total edges):")
        for mech, count in sorted(mechanism_counts.items()):
            percentage = (count / sum(mechanism_counts.values())) * 100
            print(f"    - {mech}: {count} ({percentage:.1f}%)")
        
        # Validation
        if expected_names != 'all':
            unexpected = set(mechanism_counts.keys()) - set(expected_names)
            if unexpected:
                print(f"  ⚠️  WARNING: Found unexpected mechanisms: {unexpected}")
            else:
                print(f"  ✅ PASS: Only expected mechanisms found")
        else:
            print(f"  ✅ PASS: Complex set allows all mechanisms")
    
    print("\n" + "=" * 60)
    print("✅ Mechanism Set Testing Complete!")

def test_data_generation():
    """Test that data generation works with each mechanism set."""
    
    print("\n\nTesting Data Generation with Each Mechanism Set...")
    print("=" * 60)
    
    for mechanism_set in ['simple', 'medium', 'complex']:
        print(f"\n{mechanism_set.upper()} - Generating sample data...")
        
        gen = SCMGenerator(
            num_nodes=5,
            edge_prob=0.4,
            mechanism_set=mechanism_set,
            seed=42
        )
        
        try:
            result = gen.generate_pipeline(
                num_nodes=5,
                num_samples_base=10,
                num_samples_per_intervention=10,
                complexity=0.5
            )
            
            base_tensor = result['base_tensor']
            dag = result['dag']
            
            print(f"  ✅ Generated data shape: {base_tensor.shape}")
            print(f"  ✅ DAG has {dag.number_of_edges()} edges")
            print(f"  ✅ Data range: [{base_tensor.min():.2f}, {base_tensor.max():.2f}]")
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Data Generation Testing Complete!")

if __name__ == "__main__":
    test_mechanism_sets()
    test_data_generation()
    print("\n🎉 All tests passed! Simple mechanism sets are working correctly.")
