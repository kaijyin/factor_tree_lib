#pragma once
#include "operators/baseoperator.h"

#include <string>
#include <unordered_map>
#include <xtensor/xtensor.hpp>

namespace factor_tree {
class FactorTree {
public:
  explicit FactorTree(const InitArgs &init_args);
  FactorTree(const std::string &expression, const InitArgs &init_args);
  ~FactorTree();

  void CreateTree(const std::string &expression);

  std::shared_ptr<xt::xtensor<double, 1>> Update(
      const std::unordered_map<std::string,
                               std::shared_ptr<xt::xtensor<double, 1>>> &data);

  xt::xtensor<double, 1>
  Update(const std::unordered_map<std::string, xt::xtensor<double, 1>> &data);

  std::string ToString() const { return root_->ToString(); }

  void OnDayBegin() { root_->OnDayBegin(); }
  void OnDayEnd() { root_->OnDayEnd(); }

  void SaveCheckpoint(const std::string &filename) const;

  void LoadCheckpoint(const std::string &filename);

  static std::string ParseExpression(const std::string &expression);

private:
  size_t next_req_idx_;
  std::string expression_;
  OperatorPtr root_;
  OpExprMap expr_map_;
  OperatorId next_op_id_; //   global operator id
  InitArgsPtr init_args_;
};

} // namespace factor_tree