#pragma once

#include <cereal/archives/binary.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xtensor_forward.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <glog/logging.h>

namespace factor_tree {

// 支持算子类型
enum class OperatorType : int {
  Data = 0,
  Constant,
  MathNull,
  MathRelu,
  MathAbs,
  MathLog1p,
  MathSqrt,
  MathInverse,
  MathPositive,
  MathNegative,
  MathPower2,
  MathExpm1,
  MathMinus,
  CsRank,
  CsZscore,
  CsDemean,
  CsMean,
  CsStd,
  CsSum,
  CsPosition,
  CsWinsorize,
  MathLess,
  MathGreater,
  MathAdd,
  MathSubtract,
  MathMultiply,
  MathDivide,
  MathDivide2,
  MathImbalance,
  CsOLSRes,
  TsDiff,
  TsRet,
  TsDelay,
  TsSum,
  TsMean,
  TsDemean,
  TsConcent,
  TsStd,
  TsSkew,
  TsKurt,
  TsRawSkew,
  TsRawKurt,
  TsMeanstd,
  TsMom,
  TsSlope,
  TsEma,
  TsRank,
  TsMinMax,
  TsAccelerate,
  TsMax,
  TsMin,
  TsZscore,
  TsTcorr,
  TsCorr,
  TsCov,
  TsOLSBeta,
  TsOLSAlpha,
  TsConv,
  TsYwap,
  TsTwapywap,
  TsMinMaxCps,
  TsRsi,
  TsSquareMean,
  TsAutoCorr,
  TsRs,
  TsWave,
  TsGammaalpha,
  TsGammabeta,
  TsDownvarpct,
  TsTopkMean,
  TsBotkMean,
  TsTopkStd,
  TsBotkStd,
  TsFilterTopRatio,
  TsFilterBotRatio,
  TsFilterUp,
  TsFilterDown,
  CsGroupDemean,
  CsGroupRank,
  CsGroupPosition,
  CsGroupZscore,
  CsGroupMean,
  CsGroupSum,
  CsGroupStd,
  CsMask,
  MathSymlog1p,
  MathSign,
  CsQuantilize,
  InTsMean,
  InTsStd,
  InTsEma,
  TsWmean,
  TsWstd,
  TsWskew,
  TsOlsResStd,
  TsOlsYhatStd,
  TsCoskewness,
  AdMean,
  AdSum,
};

enum class ArgType : int {
  // 目前支持四种类型的参数
  Operator = 0,
  Integer,
  Double,
  String,
};

class BaseOperator;
using OperatorId = size_t;

using Tensor = xt::xtensor<double, 1>;
using TensorPtr = std::shared_ptr<xt::xtensor<double, 1>>;

using OperatorPtr = std::shared_ptr<BaseOperator>;

using OpExprMap = std::unordered_map<std::string, OperatorPtr>;
using OpIdMap = std::unordered_map<OperatorId, OperatorPtr>;

class OpOutput {
public:
  OpOutput(TensorPtr &&data) : data_(std::move(data)) {}
  TensorPtr GetTensorPtr() { return data_; }
  Tensor &GetTensor() { return *data_; }

private:
  TensorPtr data_;
};

using RequestIdx = size_t;
class OpInput {
public:
  OpInput() = default;
  explicit OpInput(std::vector<TensorPtr> &&input_columes)
      : input_columes_(std::move(input_columes)) {}
  explicit OpInput(TensorPtr &&input_colume)
      : input_columes_({std::move(input_colume)}) {}

  inline TensorPtr GetColumeData() { return input_columes_[0]; }

  inline TensorPtr GetColumeData(int col_idx) {
    return input_columes_[col_idx];
  }

  inline TensorPtr GetLeftColumeData() { return input_columes_[0]; }
  inline TensorPtr GetRightColumeData() { return input_columes_[1]; }

private:
  std::vector<TensorPtr> input_columes_;
};

class Arg {
public:
  Arg(OperatorPtr op) : data_(op), type_(ArgType::Operator) {}
  Arg(int integer) : data_(integer), type_(ArgType::Integer) {}
  Arg(std::string str) : data_(str), type_(ArgType::String) {}
  Arg(double arg) : data_(arg), type_(ArgType::Double) {}

  OperatorPtr GetOperator() const { return std::get<OperatorPtr>(data_); }

  int GetInteger() const { return std::get<int>(data_); }

  double GetDouble() const { return std::get<double>(data_); }

  std::string GetString() const { return std::get<std::string>(data_); }

  ArgType GetType() const { return type_; }

private:
  // 确保编译器版本支持c++17特性(std::variant)
  std::variant<OperatorPtr, int, double, std::string> data_;
  ArgType type_;
};

// 初始化参数。
// 通过结构体封装,以后新加参数就不需要改原有Operator接口

struct InitArgs {
  //   nstock: 标的数量
  size_t nstock;
  //   batch_per_day: 每天的批次数量
  size_t batch_per_day;

  //   log_dir: 日志目录
  std::string log_dir;

  InitArgs() = default;
  InitArgs(const InitArgs &init_args)
      : nstock(init_args.nstock), batch_per_day(init_args.batch_per_day) {}
  InitArgs(size_t nstock) : nstock(nstock), batch_per_day(49) {}
  InitArgs(size_t nstock, size_t batch_per_day)
      : nstock(nstock), batch_per_day(batch_per_day) {}

  template <class Archive> void serialize(Archive &ar) {
    ar(nstock, batch_per_day);
  }
};

using InitArgsPtr = std::shared_ptr<InitArgs>;

struct OpInitArgs {
  OperatorId op_id;
  InitArgsPtr config;
};

class BaseOperator {
public:
  BaseOperator() = delete;
  BaseOperator(const BaseOperator &) = delete;
  virtual ~BaseOperator() = default;

  explicit BaseOperator(const OpInitArgs &init_args)
      : op_config_(init_args), current_idx_(0),
        buffer_(std::make_shared<Tensor>(
            xt::xtensor<double, 1>::from_shape({Nstock()}))) {}

  inline virtual OpOutput GetResult(RequestIdx input) = 0;

  // 获取缓冲区指针,combined op的root节点可以共用一个缓冲区,就不用拷贝了
  inline TensorPtr GetOpResultBuffer() const {
    // 注意这里永远都是值拷贝，防止被move,导致buffer指向空指针。
    // 除data其他所有op的buffer_指针在创建后就不再改变
    return buffer_;
  }

  inline size_t GetOpCacheIdx() const { return current_idx_; }

  inline void UpdateRequestIdx(RequestIdx idx) { current_idx_ = idx; }

  //   直接设置缓存数据,DataOp和CombinedOp使用
  inline void SetOpCache(size_t idx, const TensorPtr &data) {
    DCHECK((IsInputDataOp() && idx == current_idx_ + 1) || (IsCombinedOp()))
        << "current_idx_:" << current_idx_ << " input.GetRquestIdx():" << idx
        << " Nstock:" << Nstock() << " BatchPerDay:" << BatchPerDay();
    current_idx_ = idx;
    buffer_ = data;
  }

  inline size_t Nstock() const { return op_config_.config->nstock; }
  inline size_t BatchPerDay() const { return op_config_.config->batch_per_day; }

  inline const OpInitArgs &GetOpInitArgs() const { return op_config_; }

  inline const InitArgsPtr &GetInitArgs() const { return op_config_.config; }

  inline const OperatorId &GetOperatorId() const { return op_config_.op_id; }

  //   change op_id, for combined op
  inline void SetOperatorId(OperatorId op_id) { op_config_.op_id = op_id; }

  inline virtual OperatorType GetType() const = 0;

  //   是否是传输原始输入字段数据的算子,@开头
  inline bool IsInputDataOp() const { return GetType() == OperatorType::Data; }

  inline virtual bool IsCombinedOp() const { return false; }

  //   ExpOp需要实现这个接口
  inline virtual void BuildFromExpression(
      const std::function<std::pair<OperatorPtr, OpExprMap>(
          const std::string &, InitArgsPtr, OperatorId &, const OpExprMap &)>
          &build,
      OperatorId &next_op_id) {}

  inline virtual std::string ToString() const = 0;

  // 如果算子不实现这些接口，则默认不做缓存
  inline virtual void LoadCheckpoint(cereal::BinaryInputArchive &ar) {};
  inline virtual void SaveCheckpoint(cereal::BinaryOutputArchive &ar) const {};

  // 如果算子不实现这些接口，则默认不做日终处理
  inline virtual void OnDayBegin() {};
  inline virtual void OnDayEnd() {};

private:
  OpInitArgs op_config_;
  RequestIdx current_idx_;
  TensorPtr buffer_;
  std::vector<OperatorPtr> childs_;
};

struct BaseState {
  virtual void OnDayBegin() {};
  virtual void OnDayEnd() {};
};

template <typename RealOp> class UnaryOp : public BaseOperator {
public:
  UnaryOp(std::shared_ptr<BaseOperator> &child, const OpInitArgs &init_args)
      : BaseOperator(init_args), child_(child) {}
  void LoadCheckpoint(cereal::BinaryInputArchive &ar) override {
    UnaryChildLoadCheckpoint(ar);
  }

  void SaveCheckpoint(cereal::BinaryOutputArchive &ar) const override {
    UnaryChildSaveCheckpoint(ar);
  }

  void OnDayBegin() override { UnaryChildOnDayBegin(); }

  void OnDayEnd() override { UnaryChildOnDayEnd(); }

  void UnaryChildLoadCheckpoint(cereal::BinaryInputArchive &ar) {
    child_->LoadCheckpoint(ar);
  }
  void UnaryChildSaveCheckpoint(cereal::BinaryOutputArchive &ar) const {
    child_->SaveCheckpoint(ar);
  }

  void UnaryChildOnDayBegin() { child_->OnDayBegin(); }
  void UnaryChildOnDayEnd() { child_->OnDayEnd(); }

  std::shared_ptr<BaseOperator> GetChild() const { return child_; }

  //  计算函数，直接返回结果
  OpOutput GetResult(RequestIdx idx) override final {
    if (GetOpCacheIdx() == idx) {
      return OpOutput(GetOpResultBuffer());
    }
    DCHECK(idx == GetOpCacheIdx() + 1);
    auto child_output = child_->GetResult(idx);
    OpInput input{child_output.GetTensorPtr()};
    OpOutput output(GetOpResultBuffer());

    static_cast<RealOp *>(this)->Update(input, output);

    UpdateRequestIdx(idx);
    return output;
  };

private:
  std::shared_ptr<BaseOperator> child_;
};

template <typename RealOp> class BinaryOp : public BaseOperator {
public:
  BinaryOp(OperatorPtr &left_child, OperatorPtr &right_child,
           const OpInitArgs &init_args)
      : BaseOperator(init_args), left_child_(left_child),
        right_child_(right_child) {}
  void LoadCheckpoint(cereal::BinaryInputArchive &ar) override {
    BinaryChildLoadCheckpoint(ar);
  }

  void SaveCheckpoint(cereal::BinaryOutputArchive &ar) const override {
    BinaryChildSaveCheckpoint(ar);
  }

  void OnDayBegin() override { BinaryChildOnDayBegin(); }

  void OnDayEnd() override { BinaryChildOnDayEnd(); }

  void BinaryChildLoadCheckpoint(cereal::BinaryInputArchive &ar) {
    left_child_->LoadCheckpoint(ar);
    right_child_->LoadCheckpoint(ar);
  }

  void BinaryChildSaveCheckpoint(cereal::BinaryOutputArchive &ar) const {
    left_child_->SaveCheckpoint(ar);
    right_child_->SaveCheckpoint(ar);
  }

  void BinaryChildOnDayBegin() {
    left_child_->OnDayBegin();
    right_child_->OnDayBegin();
  }
  void BinaryChildOnDayEnd() {
    left_child_->OnDayEnd();
    right_child_->OnDayEnd();
  }

  // Getters for child operators
  OperatorPtr GetLeftChild() const { return left_child_; }

  OperatorPtr GetRightChild() const { return right_child_; }

  OpOutput GetResult(RequestIdx idx) override final {
    if (GetOpCacheIdx() == idx) {
      return OpOutput(GetOpResultBuffer());
    }
    DCHECK(idx == GetOpCacheIdx() + 1);
    auto left_output = left_child_->GetResult(idx);
    auto right_output = right_child_->GetResult(idx);
    std::vector<TensorPtr> input_columes{left_output.GetTensorPtr(),
                                         right_output.GetTensorPtr()};
    OpInput input(std::move(input_columes));
    OpOutput output(GetOpResultBuffer());
    //   有输入的Op需要实现这个接口
    //   即除data,constant,combined op外的所有算子都需要实现这个接口

    static_cast<RealOp *>(this)->Update(input, output);

    // 更新缓存标记为最新的idx
    UpdateRequestIdx(idx);
    return output;
  };

private:
  OperatorPtr left_child_;
  OperatorPtr right_child_;
};

template <typename State> class StateClass {
public:
  StateClass(State &&state) : state_(std::move(state)) {}
  void StateLoadCheckpoint(cereal::BinaryInputArchive &ar) { ar(state_); }
  void StateSaveCheckpoint(cereal::BinaryOutputArchive &ar) const {
    ar(state_);
  }
  void StateOnDayBegin() { state_.OnDayBegin(); }
  void StateOnDayEnd() { state_.OnDayEnd(); }

  State &GetState() { return state_; }

private:
  State state_;
};

template <typename RealOp, typename State>
class StatefulUnaryOp : public UnaryOp<RealOp>, public StateClass<State> {
public:
  using StateClass<State>::StateLoadCheckpoint;
  using StateClass<State>::StateSaveCheckpoint;
  using StateClass<State>::StateOnDayBegin;
  using StateClass<State>::StateOnDayEnd;
  using StateClass<State>::GetState;
  using UnaryOp<RealOp>::UnaryChildLoadCheckpoint;
  using UnaryOp<RealOp>::UnaryChildSaveCheckpoint;
  using UnaryOp<RealOp>::UnaryChildOnDayBegin;
  using UnaryOp<RealOp>::UnaryChildOnDayEnd;

  StatefulUnaryOp(OperatorPtr &child, State &&state,
                  const OpInitArgs &init_args)
      : UnaryOp<RealOp>(child, init_args), StateClass<State>(std::move(state)) {
  }
  // Common checkpoint handling for all stateful unary operators
  void LoadCheckpoint(cereal::BinaryInputArchive &ar) override final {
    UnaryChildLoadCheckpoint(ar);
    StateLoadCheckpoint(ar);
  }

  void SaveCheckpoint(cereal::BinaryOutputArchive &ar) const override final {
    UnaryChildSaveCheckpoint(ar);
    StateSaveCheckpoint(ar);
  }

  void OnDayBegin() override final {
    StateOnDayBegin();
    UnaryChildOnDayBegin();
  }
  void OnDayEnd() override final {
    StateOnDayEnd();
    UnaryChildOnDayEnd();
  }
};

template <typename RealOp, typename State>
class StatefulBinaryOp : public BinaryOp<RealOp>, public StateClass<State> {
public:
  using StateClass<State>::StateLoadCheckpoint;
  using StateClass<State>::StateSaveCheckpoint;
  using StateClass<State>::StateOnDayBegin;
  using StateClass<State>::StateOnDayEnd;
  using StateClass<State>::GetState;
  using BinaryOp<RealOp>::BinaryChildLoadCheckpoint;
  using BinaryOp<RealOp>::BinaryChildSaveCheckpoint;
  using BinaryOp<RealOp>::BinaryChildOnDayBegin;
  using BinaryOp<RealOp>::BinaryChildOnDayEnd;

  StatefulBinaryOp(OperatorPtr &left_child, OperatorPtr &right_child,
                   State &&state, const OpInitArgs &init_args)
      : BinaryOp<RealOp>(left_child, right_child, init_args),
        StateClass<State>(std::move(state)) {}
  void LoadCheckpoint(cereal::BinaryInputArchive &ar) override final {
    BinaryChildLoadCheckpoint(ar);
    StateLoadCheckpoint(ar);
  }

  void SaveCheckpoint(cereal::BinaryOutputArchive &ar) const override final {
    BinaryChildSaveCheckpoint(ar);
    StateSaveCheckpoint(ar);
  }

  void OnDayBegin() override final {
    StateOnDayBegin();
    BinaryChildOnDayBegin();
  }

  void OnDayEnd() override final {
    StateOnDayEnd();
    BinaryChildOnDayEnd();
  }
};

class GeneralCombOp : public BaseOperator {
public:
  GeneralCombOp(const OpInitArgs &init_args) : BaseOperator(init_args) {}
  void LoadCheckpoint(cereal::BinaryInputArchive &ar) override {
    for (auto &child : child_) {
      child.second->LoadCheckpoint(ar);
    }
    real_operator_->LoadCheckpoint(ar);
  }

  void SaveCheckpoint(cereal::BinaryOutputArchive &ar) const override {
    for (auto &child : child_) {
      child.second->SaveCheckpoint(ar);
    }
    real_operator_->SaveCheckpoint(ar);
  }

  void OnDayBegin() override {
    for (auto &child : child_) {
      child.second->OnDayBegin();
    }
    real_operator_->OnDayBegin();
  }

  void OnDayEnd() override {
    for (auto &child : child_) {
      child.second->OnDayEnd();
    }
    real_operator_->OnDayEnd();
  }

  OperatorPtr GetChild(const std::string &&child_name) const {
    auto it = child_.find(child_name);
    if (it == child_.end()) {
      throw std::invalid_argument("child_name not found");
    }
    return it->second;
  }

  void SetExpression(const std::string &expression) {
    expression_ = expression;
  };

  void SetChild(const std::string &child_name, OperatorPtr &child) {
    child_[child_name] = child;
  }

  virtual std::string GetOpExpression() const = 0;

  virtual void CombOpInit() = 0;

  bool IsCombinedOp() const override final { return true; }

  virtual void BuildFromExpression(
      const std::function<std::pair<OperatorPtr, OpExprMap>(
          const std::string &, InitArgsPtr, OperatorId &, const OpExprMap &)>
          &build,
      OperatorId &next_op_id) override final {
    CombOpInit();
    auto [root, expr_map] =
        build(expression_, GetInitArgs(), next_op_id, child_);
    real_operator_ = root;
    // root节点和当前节点使用相同的缓存空间
    SetOpCache(0, real_operator_->GetOpResultBuffer());
  }

  OpOutput GetResult(RequestIdx input) override final {
    return real_operator_->GetResult(input);
  }

private:
  // Op表达式
  std::string expression_;
  // Op计算子节点, {<"child_data1", child1>, <"child_data2", child2>}
  OpExprMap child_;
  // Op实际计算节点
  OperatorPtr real_operator_;
};

template <typename T> class UnaryCombOp : public GeneralCombOp {
public:
  UnaryCombOp(OperatorPtr &child, const OpInitArgs &init_args)
      : GeneralCombOp(init_args) {
    // 把表达式里的 @child_data 添加到基类的 child_ map 中
    SetChild("@child_data", child);
  }

  static OperatorPtr Create(const std::vector<Arg> &args,
                            const InitArgs &init_args) {
    if (args.size() != 1 || args[0].GetType() != ArgType::Operator) {
      throw std::invalid_argument(std::string(typeid(T).name()) +
                                  " operator should have 1 arguments");
    }
    auto child = args[0].GetOperator();
    return OperatorPtr(new T(child, init_args));
  }

  void CombOpInit() override final {
    std::string expression = GetOpExpression();
    SetExpression(expression);
  }

  OperatorPtr GetChildOp() const { return GetChild("@child_data"); }

  static std::vector<ArgType> ArgTypes() { return {ArgType::Operator}; }
};

template <typename T> class UnaryCombOp2 : public GeneralCombOp {
public:
  UnaryCombOp2(OperatorPtr &child, int param, const OpInitArgs &init_args)
      : GeneralCombOp(init_args), param_(param) {
    SetChild("@child_data", child);
  }
  static OperatorPtr Create(const std::vector<Arg> &args,
                            const OpInitArgs &init_args) {
    if (args.size() != 2 || args[0].GetType() != ArgType::Operator ||
        args[1].GetType() != ArgType::Integer) {
      throw std::invalid_argument(std::string(typeid(T).name()) +
                                  " operator should have 2 arguments");
    }
    auto child = args[0].GetOperator();
    int param = args[1].GetInteger();
    return OperatorPtr(new T(child, param, init_args));
  }

  static std::vector<ArgType> ArgTypes() {
    return {ArgType::Operator, ArgType::Integer};
  }

  OperatorPtr GetChildOp() const { return GetChild("@child_data"); }

  void CombOpInit() override final {
    std::string expression = std::regex_replace(
        GetOpExpression(), std::regex("\\{param\\}"), std::to_string(param_));
    SetExpression(expression);
  }

  int GetParam() const { return param_; }

private:
  int param_;
};

template <typename T> class BinaryCombOp : public GeneralCombOp {
public:
  BinaryCombOp(OperatorPtr &left_child, OperatorPtr &right_child,
               const OpInitArgs &init_args)
      : GeneralCombOp(init_args) {
    // 把表达式里的 @child_data 添加到基类的 child_ map 中
    SetChild("@child_data1", left_child);
    SetChild("@child_data2", right_child);
  }
  static OperatorPtr Create(const std::vector<Arg> &args,
                            const OpInitArgs &init_args) {
    if (args.size() != 2 || args[0].GetType() != ArgType::Operator ||
        args[1].GetType() != ArgType::Operator) {
      throw std::invalid_argument(std::string(typeid(T).name()) +
                                  " operator should have 1 arguments");
    }
    auto left_child = args[0].GetOperator();
    auto right_child = args[1].GetOperator();
    return OperatorPtr(new T(left_child, right_child, init_args));
  }

  OperatorPtr GetLeftChild() const { return GetChild("@child_data1"); }
  OperatorPtr GetRightChild() const { return GetChild("@child_data2"); }

  static std::vector<ArgType> ArgTypes() {
    return {ArgType::Operator, ArgType::Operator};
  }

  void CombOpInit() override final {
    std::string expression = GetOpExpression();
    SetExpression(expression);
  }
};

} // namespace factor_tree