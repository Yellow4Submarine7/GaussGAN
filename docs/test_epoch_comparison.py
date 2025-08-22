#!/usr/bin/env python
"""
测试逐epoch对比分析功能的简化版本
专门用于测试和验证新功能
"""

import sys
sys.path.append('/home/paperx/quantum/GaussGAN')

from compare_generators import (
    quick_status_check, 
    compare_generators_epoch_by_epoch,
    collect_all_runs_data,
    aggregate_multiple_runs,
    align_epoch_data
)

def test_data_collection():
    """测试数据收集功能"""
    print("测试1: 数据收集功能")
    print("-" * 40)
    
    data = collect_all_runs_data()
    
    if data:
        print(f"✅ 成功收集到 {len(data)} 种生成器的数据:")
        for gen_type, metrics in data.items():
            total_points = sum(len(metric_data) for metric_data in metrics.values())
            print(f"  - {gen_type}: {total_points} 个数据点")
    else:
        print("❌ 数据收集失败")
    
    return data

def test_aggregation(data):
    """测试数据聚合功能"""
    print("\n测试2: 数据聚合功能")
    print("-" * 40)
    
    aggregated = aggregate_multiple_runs(data)
    
    for gen_type, metrics in aggregated.items():
        print(f"📊 {gen_type}:")
        for metric_name, epoch_values in metrics.items():
            if epoch_values:
                print(f"  {metric_name}: {len(epoch_values)} 个epoch")
    
    return aggregated

def test_alignment(aggregated_data):
    """测试数据对齐功能"""
    print("\n测试3: 数据对齐功能")
    print("-" * 40)
    
    aligned_data, max_epochs = align_epoch_data(aggregated_data)
    
    print(f"最大epoch数: {max_epochs}")
    for gen_type, metrics in aligned_data.items():
        print(f"📊 {gen_type}:")
        for metric_name, values in metrics.items():
            valid_count = sum(1 for v in values if not np.isnan(v))
            print(f"  {metric_name}: {valid_count}/{len(values)} 个有效值")
    
    return aligned_data, max_epochs

def main():
    """主测试函数"""
    print("=" * 60)
    print("逐Epoch对比分析功能测试")
    print("=" * 60)
    
    # 检查状态
    quick_status_check()
    
    # 测试各个功能
    try:
        # 测试数据收集
        data = test_data_collection()
        if not data:
            return False
        
        # 测试聚合
        aggregated = test_aggregation(data)
        
        # 测试对齐
        aligned, max_epochs = test_alignment(aggregated)
        
        print("\n" + "=" * 60)
        print("测试结果总结")
        print("=" * 60)
        print("✅ 所有功能测试通过")
        print(f"✅ 成功处理 {len(aligned)} 种生成器")
        print(f"✅ 对齐到 {max_epochs} 个epoch")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import numpy as np
    success = main()
    
    if success:
        print("\n🎯 接下来可以运行完整分析:")
        print("python compare_generators.py")
    else:
        print("\n❌ 需要修复问题后再运行完整分析")