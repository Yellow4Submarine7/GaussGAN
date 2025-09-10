#!/usr/bin/env python3
"""
Simple script to verify if Loss charts have data
"""

import sys
sys.path.append('.')

from compare_latest import RunFinder, RunComparator

def main():
    print("🔍 Verifying Loss data loading")
    print("=" * 40)
    
    # Find latest runs
    finder = RunFinder()
    quantum_info, classical_info = finder.find_latest_runs()
    
    if not quantum_info or not classical_info:
        print("❌ Insufficient run data found")
        return
    
    print(f"✅ Quantum run: {quantum_info['generator_type']} ({quantum_info['run_id'][:8]})")
    print(f"✅ Classical run: {classical_info['generator_type']} ({classical_info['run_id'][:8]})")
    
    # Load Loss data
    comparator = RunComparator()
    quantum_losses = comparator.load_losses_from_mlflow(quantum_info)
    classical_losses = comparator.load_losses_from_mlflow(classical_info)
    
    print(f"\n📊 Loss data statistics:")
    print(f"Quantum Generator Loss: {len(quantum_losses['generator'])} data points")
    if quantum_losses['generator']:
        print(f"   Value range: {min(quantum_losses['generator']):.3f} to {max(quantum_losses['generator']):.3f}")
    
    print(f"Quantum Discriminator Loss: {len(quantum_losses['discriminator'])} data points")
    if quantum_losses['discriminator']:
        print(f"   Value range: {min(quantum_losses['discriminator']):.3f} to {max(quantum_losses['discriminator']):.3f}")
        
    print(f"Classical Generator Loss: {len(classical_losses['generator'])} data points") 
    if classical_losses['generator']:
        print(f"   Value range: {min(classical_losses['generator']):.3f} to {max(classical_losses['generator']):.3f}")
        
    print(f"Classical Discriminator Loss: {len(classical_losses['discriminator'])} data points")
    if classical_losses['discriminator']:
        print(f"   Value range: {min(classical_losses['discriminator']):.3f} to {max(classical_losses['discriminator']):.3f}")
    
    # Check if there's enough data for plotting
    has_quantum_data = len(quantum_losses['generator']) > 0 and len(quantum_losses['discriminator']) > 0
    has_classical_data = len(classical_losses['generator']) > 0 and len(classical_losses['discriminator']) > 0
    
    print(f"\n✅ Loss subplot status:")
    print(f"   Generator Loss subplot: {'Has data' if has_quantum_data and has_classical_data else '❌ Missing data'}")
    print(f"   Discriminator Loss subplot: {'Has data' if has_quantum_data and has_classical_data else '❌ Missing data'}")
    
    if has_quantum_data and has_classical_data:
        print(f"\n🎉 Fix successful! Loss subplots should now display data.")
    else:
        print(f"\n❌ Issues still need to be resolved")

if __name__ == "__main__":
    main()