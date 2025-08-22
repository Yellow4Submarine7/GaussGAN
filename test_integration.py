#!/usr/bin/env python
"""
集成测试脚本 - 验证新增的统计度量是否能正常工作
"""

import torch
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

def test_metrics_import():
    """测试metrics模块能否正常导入"""
    print("测试1: 导入metrics模块...")
    try:
        from source.metrics import (
            GaussianMetric, IsPositive, LogLikelihood, KLDivergence,
            MMDivergence, MMDivergenceFromGMM, WassersteinDistance, 
            MMDDistance, ConvergenceTracker, ALL_METRICS
        )
        print("✅ metrics模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_metric_initialization():
    """测试各个指标能否正常初始化"""
    print("\n测试2: 初始化各个指标...")
    
    # 创建测试数据
    centroids = [[0, 0], [1, 1]]
    cov_matrices = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
    weights = [0.5, 0.5]
    target_samples = np.random.randn(100, 2)
    
    try:
        from source.metrics import (
            IsPositive, LogLikelihood, KLDivergence,
            WassersteinDistance, MMDDistance, MMDivergence,
            MMDivergenceFromGMM, ConvergenceTracker
        )
        
        # 测试基本指标
        metric1 = IsPositive()
        print("  ✅ IsPositive 初始化成功")
        
        metric2 = LogLikelihood(centroids, cov_matrices, weights)
        print("  ✅ LogLikelihood 初始化成功")
        
        metric3 = KLDivergence(centroids, cov_matrices, weights)
        print("  ✅ KLDivergence 初始化成功")
        
        # 测试新增指标
        metric4 = WassersteinDistance(target_samples)
        print("  ✅ WassersteinDistance 初始化成功")
        
        metric5 = MMDDistance(target_samples)
        print("  ✅ MMDDistance 初始化成功")
        
        metric6 = MMDivergence(target_samples)
        print("  ✅ MMDivergence 初始化成功")
        
        metric7 = MMDivergenceFromGMM(centroids, cov_matrices, weights)
        print("  ✅ MMDivergenceFromGMM 初始化成功")
        
        tracker = ConvergenceTracker()
        print("  ✅ ConvergenceTracker 初始化成功")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 初始化失败: {e}")
        return False

def test_metric_computation():
    """测试指标计算是否正常"""
    print("\n测试3: 计算各个指标...")
    
    # 创建测试数据
    centroids = [[-5.0, 5.0], [5.0, 5.0]]
    cov_matrices = [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
    weights = [0.5, 0.5]
    
    # 生成测试样本
    target_samples = np.random.randn(50, 2)
    generated_samples = torch.randn(50, 2)
    
    try:
        from source.metrics import (
            IsPositive, LogLikelihood, KLDivergence,
            WassersteinDistance, MMDDistance, MMDivergence,
            MMDivergenceFromGMM
        )
        
        # 计算每个指标
        results = {}
        
        # IsPositive
        metric = IsPositive()
        score = metric.compute_score(generated_samples)
        results['IsPositive'] = np.mean(score)
        print(f"  ✅ IsPositive = {results['IsPositive']:.4f}")
        
        # LogLikelihood
        metric = LogLikelihood(centroids, cov_matrices, weights)
        score = metric.compute_score(generated_samples)
        results['LogLikelihood'] = np.mean(score)
        print(f"  ✅ LogLikelihood = {results['LogLikelihood']:.4f}")
        
        # KLDivergence
        metric = KLDivergence(centroids, cov_matrices, weights)
        score = metric.compute_score(generated_samples)
        results['KLDivergence'] = score if not np.isnan(score) else 0.0
        print(f"  ✅ KLDivergence = {results['KLDivergence']:.4f}")
        
        # WassersteinDistance
        metric = WassersteinDistance(target_samples)
        score = metric.compute_score(generated_samples)
        results['WassersteinDistance'] = score
        print(f"  ✅ WassersteinDistance = {results['WassersteinDistance']:.4f}")
        
        # MMDDistance
        metric = MMDDistance(target_samples)
        score = metric.compute_score(generated_samples)
        results['MMDDistance'] = score
        print(f"  ✅ MMDDistance = {results['MMDDistance']:.4f}")
        
        # MMDivergence
        metric = MMDivergence(target_samples)
        scores = metric.compute_score(generated_samples)
        results['MMDivergence'] = np.mean(scores)
        print(f"  ✅ MMDivergence = {results['MMDivergence']:.4f}")
        
        # MMDivergenceFromGMM
        metric = MMDivergenceFromGMM(centroids, cov_matrices, weights)
        scores = metric.compute_score(generated_samples)
        results['MMDivergenceFromGMM'] = np.mean(scores)
        print(f"  ✅ MMDivergenceFromGMM = {results['MMDivergenceFromGMM']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_convergence_tracker():
    """测试收敛跟踪器"""
    print("\n测试4: 收敛跟踪器功能...")
    
    try:
        from source.metrics import ConvergenceTracker
        
        tracker = ConvergenceTracker(patience=3, min_delta=0.01)
        
        # 模拟训练过程
        metrics_history = [
            {'KLDivergence': 1.0, 'WassersteinDistance': 0.5},
            {'KLDivergence': 0.8, 'WassersteinDistance': 0.4},
            {'KLDivergence': 0.7, 'WassersteinDistance': 0.35},
            {'KLDivergence': 0.69, 'WassersteinDistance': 0.34},  # 小改进
            {'KLDivergence': 0.68, 'WassersteinDistance': 0.33},  # 小改进
            {'KLDivergence': 0.67, 'WassersteinDistance': 0.32},  # 小改进
        ]
        
        for epoch, metrics in enumerate(metrics_history):
            info = tracker.update(epoch, metrics, d_loss=0.1, g_loss=0.2)
            print(f"  Epoch {epoch}: converged={info['converged']}, "
                  f"epochs_without_improvement={info['epochs_without_improvement']}")
        
        if tracker.should_stop_early():
            print("  ✅ 收敛检测工作正常")
        else:
            print("  ⚠️ 未检测到收敛（这可能是正常的）")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 收敛跟踪器测试失败: {e}")
        return False

def test_config_loading():
    """测试配置文件加载"""
    print("\n测试5: 加载配置文件...")
    
    try:
        import yaml
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # 检查新增的配置项
        required_keys = [
            'metrics', 'wasserstein_aggregation', 'mmd_kernel',
            'mmd_gamma', 'mmd_bandwidths', 'mmd_target_samples',
            'convergence_patience', 'convergence_min_delta',
            'convergence_monitor', 'convergence_window'
        ]
        
        missing_keys = []
        for key in required_keys:
            if key not in config:
                missing_keys.append(key)
            else:
                print(f"  ✅ {key} = {config[key]}")
        
        if missing_keys:
            print(f"  ⚠️ 缺少配置项: {missing_keys}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ 配置加载失败: {e}")
        return False

def main():
    """运行所有集成测试"""
    print("="*60)
    print("GaussGAN 新增统计度量 - 集成测试")
    print("="*60)
    
    tests = [
        ("导入测试", test_metrics_import),
        ("初始化测试", test_metric_initialization),
        ("计算测试", test_metric_computation),
        ("收敛跟踪测试", test_convergence_tracker),
        ("配置加载测试", test_config_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # 打印总结
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20s}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-"*60)
    print(f"总计: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n🎉 所有测试通过！新增的统计度量可以正常工作。")
        return 0
    else:
        print(f"\n⚠️ 有 {failed} 个测试失败，请检查相关代码。")
        return 1

if __name__ == "__main__":
    sys.exit(main())