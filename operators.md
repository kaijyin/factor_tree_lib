# Operator classification

Definition:
- ts (time series): ts ops has argument: 'window'. Inside each op, there is a default 'min_count'. 'min_count'is the inimum number of observations in window required to have a value, nan is also counted as an obsrvation. For example, ts_sum has min_count=1, so for ts_sum(x, 3), x=[1,nan,2,nan], return [1,1,3,2];
- cs (cross section)
- in (intraday): reset cache on day begin. For example, in_ts_mean(@close, 10), calculate mean of close with maximal of previous 10 observation without using previous day's data.
- ad (aggregate data): aggregate multiple days' data at the same tidx. For example, ad_mean(@volume, 10): return volumen mean of previous 10 days at the same tidx.

## Notice
- If you need to input constant data, prefix the constant with `#`.
  Example: `"add(@open,#-1.5)" or subtract(@open,#1)`
- The input expression must not contain `+`, `-`, `*`, or `/`. Instead, use `add()`, `subtract()`, `multiply()`, and `divide()` operators respectively.
- Setup double epsilon=1e-9, return nan if absolute value of denominator is less than epsilon.

## Unary

### element-wise

- [x]null(x): Return the original value of `x` without modification.
- [x]abs(x): Compute the absolute value of `x`.
- [x]sign(x): Return `1` if `x > 0`, `-1` if `x < 0`, `0` if `x == 0`.
- [x]expm1(x): Compute `e^x - 1` with higher precision for small `x`.
- [x]log1p(x): Compute `log(1 + x)` with higher precision for small `x`.
- [x]symlog1p(x): Compute `log1p(abs(x)) * sign(x)`, preserving the sign.
- [x]relu(x): Return `x` if `x > 0`, otherwise `0`.
- [x]minus(x): Compute `-x`.
- [x]inverse(x): Compute `1 / x`, return NaN if `x == 0`.
- [x]sqrt(x): Compute the square root of `x`, return NaN if `x < 0`.
- [x]power2(x): Compute `x^2`.
- [x]positive(x): Return `x` if `x >= 0`, otherwise NaN.
- [x]negative(x): Return `x` if `x < 0`, otherwise NaN.

### time series
For combined op, min_count is determined by child ops

- [x]ts_delay(x, window=1): x(t-window), return nan with not enough observations in window
- [x]ts_diff(x, window=1): x(t) - x(t-window), return nan with not enough observations in window. E.g. for ts_diff(x, 3), x=[1,3,2,4,2], return [nan,nan,nan,3,-1]
- [x]ts_ret(x, window=1): x(t) / x(t-window) - 1, return nan with not enough observations in window
- [x]ts_min(x, window=1): miminal value of sliding window, min_count=1
- [x]ts_max(x, window=1): miminal value of sliding window, min_count=1
- [x]ts_mean(x, window=1): mean value of sliding window, min_count=1
- [x]ts_ema(x, window=1): return = 2/(1+window) * x + (1 - 2/(1+window)) * old_return. If no enough observations in window, calculate eam for current observations. E.g. ts_ema(x, 4), x=[1,3,2,4,2], return [1, 0.4*3+0.6*1,...]. A forgetting window feature is setting: if 100 consecutive NaN observations are received, previous values will be forgotten.
- [x]ts_sum(x, window=1): sum value of sliding window, min_count=1
- [x]ts_std(x, window=1): std value of sliding window, min_count=2
- [x]ts_demean(x, window=1): x - ts_mean(x, windown)
- [x]ts_rank(x, window=1): get the rank value of current x in sliding windows in [0, 1] range, min_count=1
- [x]ts_zscore(x, window=1): (x-ts_mean(x, window)) / ts_std(x, window)
- [x]ts_mom(x, window=1): x / ts_mean(x, window)，Calculates momentum as the ratio of current value to its moving average, measuring relative strength against recent historical values.
- [x]ts_accelerate(x, window=1): Computes the second-order difference (acceleration) of a time series using formula x_t - 2x_{t-window} + x_{t-2window}, measuring the rate of change of change.
- [x]ts_skew(x, window=1): skewness value of sliding window, min_count=window
- [x]ts_kurt(x, window=1): kurtosis value of sliding window, min_count=window
- [x]ts_concent(x, window=1):  for ts array xs, compute concentration: tmps=abs(xs) / sum(abs(xs)), return sum(tmps**2)
- [x]ts_tcorr(x, window=1): ts_corr(x, range(window)), min_count=window
- [x]ts_rawskew(x, window=1): ts_mean(x**3, window) / ts_std(x, window)**3 raw skewness value of sliding window, min_count=window
- [x]ts_rawkurt(x, window=1): ts_mean(x**4, window) / ts_std(x, window)**4 raw kurtosis value of sliding window, min_count=window

Following are combined ops
- [x]ts_meanstd(x, window=1): ts_mean(x, window) / ts_std(x, window)
- [x]ts_squaremean(x, window=1): ts_mean(power2(x), window)
- [x]ts_autocorr(x, window=1): ts_corr(x, ts_delay(x, 1), window)
- [x]ts_rs(x, window=1): ts_mean(relu(ts_diff(x, 1)), window) / ts_mean(abs(ts_diff(x, 1)),window)
- [x]ts_rsi(x, window=1): ts_mean(relu(ts_ret(x, 1)), window) / ts_mean(abs(ts_ret(x, 1)),window)
- [x]ts_wave(x, window=1): ts_squaremean(ts_diff(x, 1), window) / ts_squaremean(x, window)
- [x]ts_gammaalpha(x, window=1): power2(ts_mean(x, window) / ts_std(x, window))
- [x]ts_gammabeta(x, window=1): ts_mean(x, window) / power2(ts_std(x, window))
- [x]ts_downvarpct: ts_squaremean(relu(x), window) / ts_squaremean(x, window)
- [x]ts_min_max_cps(x, window=1): ((ts_max(x, window) - ts_min(x, window)) / x)
- []ts_entropy(x, window=1): entropy, don't write now

### Cross section
- [x]cs_demean(x): like ts_demean, but demean in cross section
- [x]cs_rank(x): rank and scale to [-0.5, 0.5]
- [x]cs_position(x): min_max standardization
- [x]cs_mean(x): fill all non-nan value with mean(x)
- [x]cs_sum(x): fill all non-nan value with sum(x)
- [x]cs_std(x): fill all non-nan value with std(x)
- [x]cs_zscore(x): (x - cs_mean(x)) / cs_std(x)
- [x]cs_winsorize(x, std=6): winsorize x with std times standard deviation.
- [x]cs_quantilize(x, bins=10): for all non-nan value of x, group x by value. Return value is in {nan, 0, 1, ..., bins-1}. E.g. return [0,1,9,2,nan,...]

### Intraday
- [x]in_ts_mean(x, window=1): intraday version of ts_mean, reset cache each day on day begin
- [x]in_ts_std(x, window=1): intraday version of ts_std
- [x]in_ts_ema(x, window=1): intraday version of ts_ema

### aggregate
- [x]ad_mean(x, window=1): return the mean value of x at the same tidx for previous window days. Must insure the update frequency is constant each day. For instance, data frequency is 5min constantly.
- [x]ad_sum(x, window=1): similar to ad_mean

## Binary

### element-wise
For binary element-wise op(x, y), if one of x, y is nan, return nan

- less(x, y): Return 1 if x is less than y, otherwise return 0.
- greater(x, y): Return 1 if x is greater than y, otherwise return 0.
- [x]add(x, y)
- [x]subtract(x, y)
- [x]multiply(x, y)
- [x]divide(x, y): if y is very closed to 0, should return nan
- [x]divide2(x, y): symlog1p(x) - symlog1p(y), used for zero or negative y
- [x]imbalance(x, y): (x - y) / (abs(x) + abs(y))
- [x]mask(x, y): if y <= 0, return nan, else, return x.

### time series
- [x]ts_corr(x, y, window=1): pearson correlation of x, y, min_count=window
- [x]ts_cov(x, y, window=1): covariance of x, y, min_count=window
- [x]ts_ols_alpha(x, y, window=1): regress y = a + b*x + e, get a, min_count=window
- [x]ts_ols_beta(x, y, window=1): regress y = a + b*x + e, get b, min_count=window
- [x]ts_wmean(x, y, window=1): y weighted average of x, min_count=1
- [x]ts_wstd(x, y, window=1): y weighted std of x, min_count=window
- [x]ts_wskew(x, y, window=1): y weighted skewness of x, min_count=window
- [x]ts_coskewness(x, y, window=1): Coskewness of x, y , min_count=window
- [x]ts_ols_res_std(x, y, window=1): regress y = a + b*x + e, get e, then compute std(e)
- [x]ts_ols_yhat_std(x, y, window=1): regress y = a + b*x + e, get y_hat, then compute std(y_hat)
- []ts_mi(x, y, window=1): mutual information between x and y. Don't write now.
- []ts_kldiv(x, y, window=1): KL-divergency between x and y. Don't write now.

combined ops:
- [x]ts_conv(x, y, window=1): ts_mean(multiply(x, y))
- [x]ts_twapywap(x, y, window=1): ts_mean(x, window) / ts_wmean(x, y, window)

Not for factortree: ts_regresse (regression epsilon), ts_regressy (regression yhat), ts_ywtx(X, Y), ts_iywtx(X, Y) (inverse Y)


### time series filter
- [x]ts_topk_mean(X, Y, window=10, ratio=0.5): For each stock, find tidx indexes of top k=int(ratio * window) in Y, calculate mean of X[indexes], min_count=window. Nan is accepted.
- [x]ts_botk_mean(X, Y, window=10, ratio=0.5): For each stock, find tidx indexes of bot k=int(ratio * window) in Y, calculate mean of X[indexes], min_count=window. Nan is accepted.
- [x]ts_topk_std(X, Y, window=10, ratio=0.5): For each stock, find tidx indexes of top k=int(ratio * window) in Y, calculate mean of X[indexes], min_count=window. Nan is accepted.
- [x]ts_botk_std(X, Y, window=10, ratio=0.5): For each stock, find tidx indexes of bot k=int(ratio * window) in Y, calculate mean of X[indexes], min_count=window. Nan is accepted.
- [x]ts_filter_top_ratio(X, Y, window=10, ratio=0.2): For each stock, remain X based on if Y is in top 20% among previous window of Y, others return nan
- [x]ts_filter_bot_ratio(X, Y, window=10, ratio=0.2): For each stock, remain X based on if Y is in bot 20% among previous window of Y, others return nan
- [x]ts_filter_up(X, Y, window=1): remain X if Y is bigger compared to previous window value, otherwise return nan
- [x]ts_filter_down(X, Y, window=1): remain X if Y is smaller compared to previous window value, otherwise return nan

### cross section
- [x]cs_ols_res(x, y): similar to ts_ols_res, but regress y on x in cross section

Following cs ops have arguments group. group op means doing calculation inside each group. group value can take both double and nan, for group value < 0, view it as nan.
- [x]cs_group_demean(X, group): demean X inside group
- [x]cs_group_rank(X, group): rank X inside group, rank in [-0.5, 0.5]
- [x]cs_group_position(X, group): min-max scaling X inside group, position in [0, 1]
- [x]cs_group_zscore(X, group): zscore of X inside group
- [x]cs_group_mean(X, group): calculate mean of X in each group, fill the value in the same group to be the same
- [x]cs_group_sum(X, group): calculate sum of X in each group, fill the value in the same group to be the same
- [x]cs_group_std(X, group): calculate std of X in each group, fill the value in the same group to be the same

