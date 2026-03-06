import torch
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inner_product.simulator_connect import PE

def test_pe_group_mode():
    """测试PE的分组计算模式"""
    print("=== 测试PE分组计算模式 ===")
    
    # 测试数据
    weights1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    
    # 测试对角线模式
    print("\n1. 测试 diagonal 模式:")
    pe_diagonal = PE(weights=weights1, adder_num=2, name='PE_diagonal', pe_id=0, group_mode='diagonal')
    
    # 测试数据1: 所有位置都激活
    act1 = torch.ones(3, 3)
    
    output1, cycle1 = pe_diagonal.process(act1)
    expected_output1 = 1+2+3+4+5+6+7+8+9  # 45
    print(f"   所有位置激活 - 输出: {output1:.1f}, 预期: 45.0, 周期: {cycle1}")
    assert abs(output1 - expected_output1) < 1e-6, f"对角线模式测试失败: 输出 {output1} 不等于预期 {expected_output1}"
    
    # 测试数据2: 仅组0激活
    act2 = torch.zeros(3, 3)
    act2[0, 0] = 1  # 位置0
    act2[1, 1] = 1  # 位置4
    act2[2, 2] = 1  # 位置8
    output2, cycle2 = pe_diagonal.process(act2)
    expected_output2 = 1+5+9  # 15
    print(f"   仅组0激活 - 输出: {output2:.1f}, 预期: 15.0, 周期: {cycle2}")
    assert abs(output2 - expected_output2) < 1e-6, f"对角线模式测试失败: 输出 {output2} 不等于预期 {expected_output2}"
    
    # 测试数据3: 仅组1激活
    act3 = torch.zeros(3, 3)
    act3[0, 1] = 1  # 位置1
    act3[1, 2] = 1  # 位置5
    act3[2, 0] = 1  # 位置6
    output3, cycle3 = pe_diagonal.process(act3)
    expected_output3 = 2+6+7  # 15
    print(f"   仅组1激活 - 输出: {output3:.1f}, 预期: 15.0, 周期: {cycle3}")
    assert abs(output3 - expected_output3) < 1e-6, f"对角线模式测试失败: 输出 {output3} 不等于预期 {expected_output3}"
    
    # 测试数据4: 仅组2激活
    act4 = torch.zeros(3, 3)
    act4[0, 2] = 1  # 位置2
    act4[1, 0] = 1  # 位置3
    act4[2, 1] = 1  # 位置7
    output4, cycle4 = pe_diagonal.process(act4)
    expected_output4 = 3+4+8  # 15
    print(f"   仅组2激活 - 输出: {output4:.1f}, 预期: 15.0, 周期: {cycle4}")
    assert abs(output4 - expected_output4) < 1e-6, f"对角线模式测试失败: 输出 {output4} 不等于预期 {expected_output4}"
    
    # 测试行模式
    print("\n2. 测试 row 模式:")
    pe_row = PE(weights=weights1, adder_num=2, name='PE_row', pe_id=1, group_mode='row')
    
    # 测试数据1: 所有位置都激活
    output5, cycle5 = pe_row.process(act1)
    expected_output5 = 45
    print(f"   所有位置激活 - 输出: {output5:.1f}, 预期: 45.0, 周期: {cycle5}")
    assert abs(output5 - expected_output5) < 1e-6, f"行模式测试失败: 输出 {output5} 不等于预期 {expected_output5}"
    
    # 测试数据2: 仅组0激活（第一行）
    act6 = torch.zeros(3, 3)
    act6[0, 0] = 1  # 位置0
    act6[0, 1] = 1  # 位置1
    act6[0, 2] = 1  # 位置2
    output6, cycle6 = pe_row.process(act6)
    expected_output6 = 1+2+3  # 6
    print(f"   仅组0激活 - 输出: {output6:.1f}, 预期: 6.0, 周期: {cycle6}")
    assert abs(output6 - expected_output6) < 1e-6, f"行模式测试失败: 输出 {output6} 不等于预期 {expected_output6}"
    
    # 测试数据3: 仅组1激活（第二行）
    act7 = torch.zeros(3, 3)
    act7[1, 0] = 1  # 位置3
    act7[1, 1] = 1  # 位置4
    act7[1, 2] = 1  # 位置5
    output7, cycle7 = pe_row.process(act7)
    expected_output7 = 4+5+6  # 15
    print(f"   仅组1激活 - 输出: {output7:.1f}, 预期: 15.0, 周期: {cycle7}")
    assert abs(output7 - expected_output7) < 1e-6, f"行模式测试失败: 输出 {output7} 不等于预期 {expected_output7}"
    
    # 测试数据4: 仅组2激活（第三行）
    act8 = torch.zeros(3, 3)
    act8[2, 0] = 1  # 位置6
    act8[2, 1] = 1  # 位置7
    act8[2, 2] = 1  # 位置8
    output8, cycle8 = pe_row.process(act8)
    expected_output8 = 7+8+9  # 24
    print(f"   仅组2激活 - 输出: {output8:.1f}, 预期: 24.0, 周期: {cycle8}")
    assert abs(output8 - expected_output8) < 1e-6, f"行模式测试失败: 输出 {output8} 不等于预期 {expected_output8}"
    
    # 测试边界情况
    print("\n3. 测试边界情况:")
    
    # 全零激活
    act_zero = torch.zeros(3, 3)
    output_zero, cycle_zero = pe_diagonal.process(act_zero)
    print(f"   全零激活 - 输出: {output_zero:.1f}, 周期: {cycle_zero}")
    assert abs(output_zero) < 1e-6, f"全零激活测试失败: 输出 {output_zero} 不为零"
    
    # 单个激活
    act_single = torch.zeros(3, 3)
    act_single[0, 0] = 1
    output_single, cycle_single = pe_diagonal.process(act_single)
    print(f"   单个激活 - 输出: {output_single:.1f}, 周期: {cycle_single}")
    assert abs(output_single - 1.0) < 1e-6, f"单个激活测试失败: 输出 {output_single} 不等于预期 1.0"
    
    # 测试1的数目比较多的情况（高密度激活）
    print("\n4. 测试1的数目比较多的情况（高密度激活）:")
    
    # 情况1: 8个位置激活（只有1个位置为0）
    act_dense1 = torch.ones(3, 3)
    act_dense1[2, 2] = 0  # 只有一个位置为0
    output_dense1, cycle_dense1 = pe_diagonal.process(act_dense1)
    expected_dense1 = 1+2+3+4+5+6+7+8  # 36
    print(f"   8个位置激活 - 输出: {output_dense1:.1f}, 预期: {expected_dense1}, 周期: {cycle_dense1}")
    assert abs(output_dense1 - expected_dense1) < 1e-6, f"高密度激活测试失败: 输出 {output_dense1} 不等于预期 {expected_dense1}"
    
    # 情况2: 7个位置激活（2个位置为0）
    act_dense2 = torch.ones(3, 3)
    act_dense2[1, 1] = 0
    act_dense2[2, 2] = 0
    output_dense2, cycle_dense2 = pe_diagonal.process(act_dense2)
    expected_dense2 = 1+2+3+4+6+7+8  # 31
    print(f"   7个位置激活 - 输出: {output_dense2:.1f}, 预期: {expected_dense2}, 周期: {cycle_dense2}")
    assert abs(output_dense2 - expected_dense2) < 1e-6, f"高密度激活测试失败: 输出 {output_dense2} 不等于预期 {expected_dense2}"
    
    # 情况3: 6个位置激活（3个位置为0）
    act_dense3 = torch.ones(3, 3)
    act_dense3[0, 0] = 0
    act_dense3[1, 1] = 0
    act_dense3[2, 2] = 0
    output_dense3, cycle_dense3 = pe_diagonal.process(act_dense3)
    expected_dense3 = 2+3+4+6+7+8  # 30
    print(f"   6个位置激活 - 输出: {output_dense3:.1f}, 预期: {expected_dense3}, 周期: {cycle_dense3}")
    assert abs(output_dense3 - expected_dense3) < 1e-6, f"高密度激活测试失败: 输出 {output_dense3} 不等于预期 {expected_dense3}"
    
    # 测试不均衡的情况（某些组激活多，某些组激活少）
    print("\n5. 测试不均衡的情况（组间激活数量差异大）:")
    
    # 对角线模式 - 组0有3个，组1有2个，组2有0个
    act_imbalanced1 = torch.zeros(3, 3)
    # 组0: 位置0, 4, 8
    act_imbalanced1[0, 0] = 1
    act_imbalanced1[1, 1] = 1
    act_imbalanced1[2, 2] = 1
    # 组1: 位置1, 5
    act_imbalanced1[0, 1] = 1
    act_imbalanced1[1, 2] = 1
    # 组2: 无激活
    output_imbalanced1, cycle_imbalanced1 = pe_diagonal.process(act_imbalanced1)
    expected_imbalanced1 = 1+2+5+6+9  # 23
    print(f"   对角线模式 - 组0:3个, 组1:2个, 组2:0个 - 输出: {output_imbalanced1:.1f}, 预期: {expected_imbalanced1}, 周期: {cycle_imbalanced1}")
    assert abs(output_imbalanced1 - expected_imbalanced1) < 1e-6, f"不均衡测试失败: 输出 {output_imbalanced1} 不等于预期 {expected_imbalanced1}"
    
    # 对角线模式 - 组0有0个，组1有3个，组2有3个
    act_imbalanced2 = torch.zeros(3, 3)
    # 组0: 无激活
    # 组1: 位置1, 5, 6
    act_imbalanced2[0, 1] = 1
    act_imbalanced2[1, 2] = 1
    act_imbalanced2[2, 0] = 1
    # 组2: 位置2, 3, 7
    act_imbalanced2[0, 2] = 1
    act_imbalanced2[1, 0] = 1
    act_imbalanced2[2, 1] = 1
    output_imbalanced2, cycle_imbalanced2 = pe_diagonal.process(act_imbalanced2)
    expected_imbalanced2 = 2+3+4+6+7+8  # 30
    print(f"   对角线模式 - 组0:0个, 组1:3个, 组2:3个 - 输出: {output_imbalanced2:.1f}, 预期: {expected_imbalanced2}, 周期: {cycle_imbalanced2}")
    assert abs(output_imbalanced2 - expected_imbalanced2) < 1e-6, f"不均衡测试失败: 输出 {output_imbalanced2} 不等于预期 {expected_imbalanced2}"
    
    # 行模式 - 组0有3个，组1有1个，组2有0个
    act_imbalanced3 = torch.zeros(3, 3)
    # 组0: 第一行全部
    act_imbalanced3[0, 0] = 1
    act_imbalanced3[0, 1] = 1
    act_imbalanced3[0, 2] = 1
    # 组1: 位置4
    act_imbalanced3[1, 1] = 1
    # 组2: 无激活
    output_imbalanced3, cycle_imbalanced3 = pe_row.process(act_imbalanced3)
    expected_imbalanced3 = 1+2+3+5  # 11
    print(f"   行模式 - 组0:3个, 组1:1个, 组2:0个 - 输出: {output_imbalanced3:.1f}, 预期: {expected_imbalanced3}, 周期: {cycle_imbalanced3}")
    assert abs(output_imbalanced3 - expected_imbalanced3) < 1e-6, f"不均衡测试失败: 输出 {output_imbalanced3} 不等于预期 {expected_imbalanced3}"
    
    # 行模式 - 组0有0个，组1有0个，组2有3个
    act_imbalanced4 = torch.zeros(3, 3)
    # 组0: 无激活
    # 组1: 无激活
    # 组2: 第三行全部
    act_imbalanced4[2, 0] = 1
    act_imbalanced4[2, 1] = 1
    act_imbalanced4[2, 2] = 1
    output_imbalanced4, cycle_imbalanced4 = pe_row.process(act_imbalanced4)
    expected_imbalanced4 = 7+8+9  # 24
    print(f"   行模式 - 组0:0个, 组1:0个, 组2:3个 - 输出: {output_imbalanced4:.1f}, 预期: {expected_imbalanced4}, 周期: {cycle_imbalanced4}")
    assert abs(output_imbalanced4 - expected_imbalanced4) < 1e-6, f"不均衡测试失败: 输出 {output_imbalanced4} 不等于预期 {expected_imbalanced4}"
    
    # 极端不均衡情况：所有激活都集中在一个组
    print("\n6. 测试极端不均衡情况（所有激活集中在一个组）:")
    
    # 对角线模式 - 所有激活都在组0
    act_extreme1 = torch.zeros(3, 3)
    act_extreme1[0, 0] = 1  # 组0
    act_extreme1[1, 1] = 1  # 组0
    act_extreme1[2, 2] = 1  # 组0
    output_extreme1, cycle_extreme1 = pe_diagonal.process(act_extreme1)
    expected_extreme1 = 1+5+9  # 15
    print(f"   对角线模式 - 所有激活在组0 - 输出: {output_extreme1:.1f}, 预期: {expected_extreme1}, 周期: {cycle_extreme1}")
    assert abs(output_extreme1 - expected_extreme1) < 1e-6, f"极端不均衡测试失败: 输出 {output_extreme1} 不等于预期 {expected_extreme1}"
    
    # 行模式 - 所有激活都在组1
    act_extreme2 = torch.zeros(3, 3)
    act_extreme2[1, 0] = 1  # 组1
    act_extreme2[1, 1] = 1  # 组1
    act_extreme2[1, 2] = 1  # 组1
    output_extreme2, cycle_extreme2 = pe_row.process(act_extreme2)
    expected_extreme2 = 4+5+6  # 15
    print(f"   行模式 - 所有激活在组1 - 输出: {output_extreme2:.1f}, 预期: {expected_extreme2}, 周期: {cycle_extreme2}")
    assert abs(output_extreme2 - expected_extreme2) < 1e-6, f"极端不均衡测试失败: 输出 {output_extreme2} 不等于预期 {expected_extreme2}"
    
    print("\n✅ 所有测试通过！")

if __name__ == "__main__":
    test_pe_group_mode()
