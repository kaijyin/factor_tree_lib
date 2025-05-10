#include "factor_tree/factortree.h"
#include <iomanip>
#include <iostream>
#include <random>

int main() {
  std::cout << "FactorTree C++ Demo" << std::endl;
  std::cout << "===================" << std::endl;

  // 初始化参数
  const size_t STOCK_COUNT = 5;
  InitArgs args(STOCK_COUNT);
  args.batch_per_day = 40;

  std::cout << "创建计算树: " << STOCK_COUNT << "只股票，每日"
            << args.batch_per_day << "批数据" << std::endl;

  // 创建FactorTree实例
  FactorTree factor_tree(args);

  // 创建计算表达式: 5日移动平均
  std::string expression = "ts_mean(@open, 5)";
  factor_tree.CreateTree(expression);
  std::cout << "计算表达式: " << expression << std::endl;
  std::cout << "树结构: " << factor_tree.ToString() << std::endl;

  // 随机数生成器
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(50.0, 150.0);

  // 模拟计算3天数据
  for (int day = 0; day < 3; ++day) {
    factor_tree.OnDayBegin();
    std::cout << "\n第 " << day + 1 << " 天开始" << std::endl;

    // 每天处理4批数据
    for (int batch = 0; batch < 4; ++batch) {
      // 准备输入数据 - 5只股票的开盘价
      xt::xtensor<double, 1> open_prices = xt::zeros<double>({STOCK_COUNT});
      for (size_t i = 0; i < STOCK_COUNT; ++i) {
        open_prices(i) = dis(gen);
      }

      // 准备输入数据字典 - 使用构造函数而不是赋值
      std::unordered_map<std::string, const xt::xtensor<double, 1>> data;
      data.emplace("open", open_prices); // 使用emplace而不是赋值

      // 更新计算树并获取结果
      xt::xtensor<double, 1> result = factor_tree.Update(data);

      // 显示输入和输出
      std::cout << "  批次 " << batch + 1 << ":" << std::endl;
      std::cout << "  输入: ";
      for (size_t i = 0; i < STOCK_COUNT; ++i) {
        std::cout << std::fixed << std::setprecision(2) << open_prices(i)
                  << " ";
      }
      std::cout << std::endl;

      std::cout << "  输出: ";
      for (size_t i = 0; i < STOCK_COUNT; ++i) {
        std::cout << std::fixed << std::setprecision(2);
        if (std::isnan(result(i))) {
          std::cout << "NaN ";
        } else {
          std::cout << result(i) << " ";
        }
      }
      std::cout << std::endl;
    }

    factor_tree.OnDayEnd();
    std::cout << "第 " << day + 1 << " 天结束" << std::endl;
  }

  // 保存检查点
  std::string checkpoint_file = "factor_tree_checkpoint.bin";
  factor_tree.SaveCheckpoint(checkpoint_file);
  std::cout << "\n检查点已保存到: " << checkpoint_file << std::endl;

  // 从检查点加载
  FactorTree new_tree(args);
  new_tree.LoadCheckpoint(checkpoint_file);
  std::cout << "从检查点加载到新树，表达式: " << new_tree.ToString()
            << std::endl;

  std::cout << "\n演示完成！" << std::endl;
  return 0;
}