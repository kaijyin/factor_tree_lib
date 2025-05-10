"""
FactorTree Python 演示程序

这个脚本演示了如何使用 factor_tree_py 模块进行金融时间序列数据的因子计算。
包括创建因子计算树、更新数据、处理交易日循环以及保存/加载检查点等功能。
"""

import numpy as np
import factor_tree_py as ft


def main() -> None:
    """
    FactorTree 主演示函数

    演示 FactorTree 的基本功能，包括：
    1. 初始化参数和计算树
    2. 创建"5日移动平均"因子表达式
    3. 模拟3个交易日的数据处理循环
    4. 保存和加载检查点文件

    无需输入参数，直接运行即可查看演示效果。
    """
    print("FactorTree Python Demo")
    print("=====================")

    # 初始化参数
    nstock = 5
    init_args = ft.InitArgs(nstock, 40)

    print(f"创建计算树: {nstock}只股票，每日{init_args.batch_per_day}批数据")

    # 创建FactorTree实例
    factor_tree = ft.FactorTree(init_args)

    # 创建计算表达式: 5日移动平均
    expression = "ts_mean(@open, 5)"
    factor_tree.create_tree(expression)
    print(f"计算表达式: {expression}")
    print(f"树结构: {factor_tree.to_string()}")

    # 模拟计算3天数据
    for day in range(3):
        factor_tree.on_day_begin()
        print(f"\n第 {day + 1} 天开始")

        # 每天处理4批数据
        for batch in range(4):
            # 准备输入数据 - 5只股票的开盘价
            open_prices = np.random.uniform(50.0, 150.0, nstock)

            # 准备输入数据字典
            data = {"open": open_prices}

            # 更新计算树并获取结果
            result = factor_tree.update(data)

            # 显示输入和输出
            print(f"  批次 {batch + 1}:")
            print(f"  输入: {' '.join([f'{x:.2f}' for x in open_prices])}")

            # 处理可能的NaN值
            output_str = []
            for val in result:
                if np.isnan(val):
                    output_str.append("NaN")
                else:
                    output_str.append(f"{val:.2f}")
            print(f"  输出: {' '.join(output_str)}")

        factor_tree.on_day_end()
        print(f"第 {day + 1} 天结束")

    # 保存检查点
    checkpoint_file = "factor_tree_checkpoint_py.bin"
    factor_tree.save_checkpoint(checkpoint_file)
    print(f"\n检查点已保存到: {checkpoint_file}")

    # 从检查点加载
    new_tree = ft.FactorTree(init_args)
    new_tree.load_checkpoint(checkpoint_file)
    print(f"从检查点加载到新树，表达式: {new_tree.to_string()}")

    print("\n演示完成！")


if __name__ == "__main__":
    main()
