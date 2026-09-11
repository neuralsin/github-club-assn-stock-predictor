import numpy as np
from scipy.stats import norm

def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    ret = np.asarray(returns)
    ret = ret[~np.isnan(ret)]
    if len(ret) == 0 or ret.std() == 0:
        return 0.0
    return float((ret.mean() / ret.std()) * np.sqrt(periods_per_year))

def deflated_sharpe_ratio(
    observed_sr: float,
    sr_trials: list[float],
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    trials = [s for s in sr_trials if not np.isnan(s)]
    n_trials = len(trials)
    if n_trials <= 1 or n_obs <= 1:
        return 0.5

    sr_std_trials = float(np.std(trials)) if np.std(trials) > 0 else 1e-6
    euler_mascheroni = 0.5772156649

    term1 = (1.0 - euler_mascheroni) * norm.ppf(1.0 - 1.0 / n_trials)
    term2 = euler_mascheroni * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    expected_max_sr = sr_std_trials * (term1 + term2)

    denom = max(1, n_obs - 1)
    variance_term = (1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * (observed_sr ** 2)) / denom
    sr_std = np.sqrt(max(1e-9, variance_term))

    z_score = (observed_sr - expected_max_sr) / sr_std
    return float(norm.cdf(z_score))

