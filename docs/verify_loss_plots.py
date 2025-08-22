#!/usr/bin/env python3
"""
验证Loss图表是否有数据的简单脚本
"""

import sys
sys.path.append('.')

from compare_latest import RunFinder, RunComparator

def main():
    print("🔍 验证Loss数据加载")
    print("=" * 40)
    
    # 查找最新运行
    finder = RunFinder()
    quantum_info, classical_info = finder.find_latest_runs()
    
    if not quantum_info or not classical_info:
        print("❌ 未找到足够的运行数据")
        return
    
    print(f"✅ 量子运行: {quantum_info['generator_type']} ({quantum_info['run_id'][:8]})")
    print(f"✅ 经典运行: {classical_info['generator_type']} ({classical_info['run_id'][:8]})")
    
    # 加载Loss数据
    comparator = RunComparator()
    quantum_losses = comparator.load_losses_from_mlflow(quantum_info)
    classical_losses = comparator.load_losses_from_mlflow(classical_info)
    
    print(f"\n📊 Loss数据统计:")
    print(f"量子Generator Loss: {len(quantum_losses['generator'])} 数据点")
    if quantum_losses['generator']:
        print(f"   数值范围: {min(quantum_losses['generator']):.3f} 到 {max(quantum_losses['generator']):.3f}")
    
    print(f"量子Discriminator Loss: {len(quantum_losses['discriminator'])} 数据点")
    if quantum_losses['discriminator']:
        print(f"   数值范围: {min(quantum_losses['discriminator']):.3f} 到 {max(quantum_losses['discriminator']):.3f}")
        
    print(f"经典Generator Loss: {len(classical_losses['generator'])} 数据点") 
    if classical_losses['generator']:
        print(f"   数值范围: {min(classical_losses['generator']):.3f} 到 {max(classical_losses['generator']):.3f}")
        
    print(f"经典Discriminator Loss: {len(classical_losses['discriminator'])} 数据点")
    if classical_losses['discriminator']:
        print(f"   数值范围: {min(classical_losses['discriminator']):.3f} 到 {max(classical_losses['discriminator']):.3f}")
    
    # 检查是否有足够的数据用于绘图
    has_quantum_data = len(quantum_losses['generator']) > 0 and len(quantum_losses['discriminator']) > 0
    has_classical_data = len(classical_losses['generator']) > 0 and len(classical_losses['discriminator']) > 0
    
    print(f"\n✅ Loss子图状态:")
    print(f"   Generator Loss子图: {'有数据' if has_quantum_data and has_classical_data else '❌ 缺少数据'}")
    print(f"   Discriminator Loss子图: {'有数据' if has_quantum_data and has_classical_data else '❌ 缺少数据'}")
    
    if has_quantum_data and has_classical_data:
        print(f"\n🎉 修复成功！Loss子图现在应该显示数据了。")
    else:
        print(f"\n❌ 仍有问题需要解决")

if __name__ == "__main__":
    main()