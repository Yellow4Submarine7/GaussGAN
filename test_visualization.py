#!/usr/bin/env python3
"""
Test script to demonstrate the epoch-by-epoch visualization functionality.
This script runs a few epochs for different generator types to show the visualization works.
"""

import subprocess
import sys

def run_training(generator_type, epochs=3):
    """Run training for a specific generator type with given epochs."""
    print(f"\n{'='*60}")
    print(f"Testing {generator_type} generator for {epochs} epochs")
    print(f"{'='*60}")
    
    cmd = [
        "uv", "run", "python", "main.py",
        "--generator_type", generator_type,
        "--max_epochs", str(epochs)
    ]
    
    try:
        # Run with timeout to prevent hanging
        result = subprocess.run(
            cmd,
            timeout=300,  # 5 minutes timeout
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ Successfully completed training for {generator_type}")
            # Extract visualization messages from stdout
            lines = result.stdout.split('\n')
            viz_lines = [line for line in lines if 'Visualization saved to' in line]
            for line in viz_lines:
                print(f"  {line}")
        else:
            print(f"❌ Training failed for {generator_type}")
            print(f"Error: {result.stderr[-500:]}")  # Last 500 chars of error
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Training timed out for {generator_type}")
    except Exception as e:
        print(f"💥 Exception occurred for {generator_type}: {e}")

def main():
    """Main test function."""
    print("GaussGAN Epoch-by-Epoch Visualization Test")
    print("This script will test visualization for different generator types")
    
    # Test different generator types
    generator_types = [
        "classical_normal",
        "classical_uniform", 
        "quantum_samples",
        "quantum_shadows"
    ]
    
    print(f"\nWill test the following generator types: {', '.join(generator_types)}")
    print("Each will run for 2 epochs to demonstrate the visualization functionality.")
    
    # Ask for confirmation
    try:
        response = input("\nProceed with testing? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Testing cancelled.")
            return
    except KeyboardInterrupt:
        print("\nTesting cancelled.")
        return
    
    # Run tests
    for generator_type in generator_types:
        run_training(generator_type, epochs=2)
    
    print(f"\n{'='*60}")
    print("Testing Complete!")
    print("Check the images/ directory for the generated visualizations:")
    print("  training_0_classical_normal.png")
    print("  training_1_classical_normal.png")
    print("  training_0_classical_uniform.png") 
    print("  training_1_classical_uniform.png")
    print("  training_0_quantum_samples.png")
    print("  training_1_quantum_samples.png")
    print("  training_0_quantum_shadows.png")
    print("  training_1_quantum_shadows.png")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()