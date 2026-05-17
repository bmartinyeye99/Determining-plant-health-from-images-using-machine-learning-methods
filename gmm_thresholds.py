"""
Shared GMM threshold utilities — single source of truth.

Used by:
  - eda.py                 (NDVI thresholds for label generation)
  - baseline_indices.py    (ExG / VARI / NGRDI thresholds for RGB baselines)
  - cross_domain_eval.py   (per-target-domain re-fitting for any index)

Why this module exists:
  Previously each of the three files had its own copy of the GMM fit +
  threshold computation. That meant the same bug (using the midpoint of
  adjacent component means instead of the Bayes-optimal MAP boundary)
  appeared in three places, and any future fix had to be made three times.
  All threshold-derivation logic now lives here.

Method:
  We fit a 4-component Gaussian Mixture Model to the index distribution
  (typically NDVI). Each component k has parameters (w_k, mu_k, sigma_k)
  representing prior weight, mean, and standard deviation respectively.

  For a new sample x, the posterior probability of belonging to component k is
                          w_k * N(x | mu_k, sigma_k^2)
        P(k | x)  =  ─────────────────────────────────────
                      sum_j w_j * N(x | mu_j, sigma_j^2)

  The maximum-a-posteriori (MAP) classifier picks the component with the
  largest numerator. The decision boundary between two adjacent (sorted)
  components k and k+1 is therefore the x at which
        w_k * N(x | mu_k, sigma_k^2)  =  w_{k+1} * N(x | mu_{k+1}, sigma_{k+1}^2).
  Taking logs and rearranging yields a quadratic in x:
        a x^2 + b x + c = 0
  with
        a =  1/(2 sigma_k^2)  -  1/(2 sigma_{k+1}^2)
        b =  mu_{k+1}/sigma_{k+1}^2  -  mu_k/sigma_k^2
        c =  mu_k^2/(2 sigma_k^2)  -  mu_{k+1}^2/(2 sigma_{k+1}^2)
            + log(sigma_k / sigma_{k+1})  -  log(w_k / w_{k+1}).
  The Bayes-optimal threshold is the root that lies between mu_k and mu_{k+1}.

  The midpoint formula (mu_k + mu_{k+1})/2 only equals this root when the
  two components have identical variance AND identical weight. In real
  vegetation NDVI distributions, neither is true: the dead/soil cluster is
  typically much wider than the healthy cluster, and the class priors differ
  by 2-3x, so the midpoint approximation systematically misplaces the
  boundary.

Reference:
  Bishop, C. M. (2006). "Pattern Recognition and Machine Learning",
  Section 4.2 (Probabilistic Generative Models).
"""

import numpy as np
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# Bayes-optimal crossing point between two weighted Gaussians
# ---------------------------------------------------------------------------
def gaussian_crossing(mu1, sigma1, w1, mu2, sigma2, w2):
    """Return the MAP decision boundary between two weighted Gaussians.

    Solves  w1 * N(x | mu1, sigma1^2) = w2 * N(x | mu2, sigma2^2)
    and returns the root that lies between mu1 and mu2.

    Args:
        mu1, mu2:         Component means (floats). Order does not matter.
        sigma1, sigma2:   Component standard deviations (positive floats).
        w1, w2:           Component prior weights (positive floats; need not
                          sum to 1 since only their ratio enters the equation).

    Returns:
        Float. The MAP decision boundary between the two components.
        Falls back to the midpoint (mu1 + mu2)/2 only in pathological cases
        where the quadratic has no real root in [min(mu1,mu2), max(mu1,mu2)].
    """
    # Equal-variance edge case: quadratic collapses to linear.
    if abs(sigma1 - sigma2) < 1e-12:
        # Linear solution: x = (mu1 + mu2)/2 - sigma^2/(mu2 - mu1) * log(w1/w2)
        if abs(mu2 - mu1) < 1e-12:
            return (mu1 + mu2) / 2.0  # degenerate
        return (mu1 + mu2) / 2.0 \
               - (sigma1 ** 2) / (mu2 - mu1) * np.log(w1 / w2)

    # General quadratic case
    a = 1.0 / (2 * sigma1 ** 2) - 1.0 / (2 * sigma2 ** 2)
    b = mu2 / (sigma2 ** 2) - mu1 / (sigma1 ** 2)
    c = (mu1 ** 2) / (2 * sigma1 ** 2) - (mu2 ** 2) / (2 * sigma2 ** 2) \
        + np.log(sigma1 / sigma2) - np.log(w1 / w2)

    disc = b ** 2 - 4 * a * c
    if disc < 0:
        # No real crossing — components don't actually cross in PDF space.
        # This happens only when one component completely dominates.
        return (mu1 + mu2) / 2.0

    sqrt_disc = np.sqrt(disc)
    r1 = (-b + sqrt_disc) / (2 * a)
    r2 = (-b - sqrt_disc) / (2 * a)

    lo, hi = min(mu1, mu2), max(mu1, mu2)
    candidates = [r for r in (r1, r2) if lo <= r <= hi]
    if len(candidates) == 1:
        return float(candidates[0])
    if len(candidates) == 2:
        # Both roots in interval (rare; pick the one closer to weighted mean)
        weighted_mean = (w1 * mu1 + w2 * mu2) / (w1 + w2)
        return float(min(candidates, key=lambda r: abs(r - weighted_mean)))

    # Fallback: no root inside [mu1, mu2] — usually a degenerate fit.
    return (mu1 + mu2) / 2.0


# ---------------------------------------------------------------------------
# Fit a 4-component GMM and return (thresholds, gmm, sort_index)
# ---------------------------------------------------------------------------
def fit_gmm_4class(values_1d, n_init=3, random_state=42, max_iter=300):
    """Fit a 4-component GMM on a 1-D array and return MAP thresholds.

    Args:
        values_1d:     1-D numpy array of finite index values
                       (e.g. NDVI, ExG, VARI, NGRDI samples).
        n_init:        Number of EM restarts for stability. Default 3.
        random_state:  RNG seed for reproducibility. Default 42.
        max_iter:      Maximum EM iterations per restart. Default 300.

    Returns:
        thresholds:  list of 3 floats, ascending, MAP-optimal boundaries.
        gmm:         fitted GaussianMixture object.
        sort_idx:    np.ndarray such that gmm.means_.flatten()[sort_idx] is
                     sorted ascending. Use this to align means_, covariances_,
                     and weights_ to the same ordering used by `thresholds`.
    """
    data_2d = np.asarray(values_1d, dtype=np.float64).reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=4,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
    ).fit(data_2d)

    means = gmm.means_.flatten()
    sort_idx = np.argsort(means)

    sorted_means = means[sort_idx]
    sorted_stds = np.sqrt(gmm.covariances_.flatten()[sort_idx])
    sorted_weights = gmm.weights_[sort_idx]

    thresholds = []
    for i in range(3):
        t = gaussian_crossing(
            sorted_means[i], sorted_stds[i], sorted_weights[i],
            sorted_means[i + 1], sorted_stds[i + 1], sorted_weights[i + 1],
        )
        thresholds.append(float(t))

    return thresholds, gmm, sort_idx


# ---------------------------------------------------------------------------
# BIC / AIC sweep for model-selection diagnostics
# ---------------------------------------------------------------------------
def gmm_model_selection(values_1d, k_range=range(2, 7),
                        n_init=3, random_state=42, max_iter=300):
    """Fit GMMs with k components in k_range and report BIC and AIC.

    Used in the thesis methods chapter to justify the k=4 choice. If BIC
    prefers a different k, document it explicitly: "BIC favored k=X but we
    constrain k=4 to align with the four-tier vegetation taxonomy."

    Args:
        values_1d:     1-D numpy array of index values.
        k_range:       Iterable of component counts to evaluate.
        n_init, random_state, max_iter: Passed to GaussianMixture.

    Returns:
        bic_scores:    dict {k: bic}
        aic_scores:    dict {k: aic}
        fitted_gmms:   dict {k: fitted GaussianMixture}
    """
    data_2d = np.asarray(values_1d, dtype=np.float64).reshape(-1, 1)
    bic_scores, aic_scores, fitted_gmms = {}, {}, {}
    for k in k_range:
        gmm_k = GaussianMixture(
            n_components=k, random_state=random_state,
            max_iter=max_iter, n_init=n_init,
        ).fit(data_2d)
        bic_scores[k] = gmm_k.bic(data_2d)
        aic_scores[k] = gmm_k.aic(data_2d)
        fitted_gmms[k] = gmm_k
    return bic_scores, aic_scores, fitted_gmms


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """Sanity check on a synthetic 4-Gaussian mixture."""
    rng = np.random.RandomState(0)
    samples = np.concatenate([
        rng.normal(-0.30, 0.15, 4000),   # dead/soil  - wide, common
        rng.normal( 0.05, 0.08, 2500),   # severe     - narrower, less common
        rng.normal( 0.35, 0.06, 2000),   # moderate   - narrow
        rng.normal( 0.65, 0.05, 1500),   # healthy    - tightest, rarest
    ])

    thresholds, gmm, idx = fit_gmm_4class(samples)
    means = gmm.means_.flatten()[idx]
    stds = np.sqrt(gmm.covariances_.flatten()[idx])
    weights = gmm.weights_[idx]

    print("Component fits (sorted by mean):")
    for i, name in enumerate(["dead/soil", "severe", "moderate", "healthy"]):
        print(f"  {name:>10s}: mu={means[i]:+.4f}  sigma={stds[i]:.4f}  w={weights[i]:.3f}")

    midpoints = [(means[i] + means[i + 1]) / 2 for i in range(3)]
    print("\nThreshold comparison:")
    print(f"  {'boundary':<16} {'midpoint':>10}  {'MAP':>10}  {'shift':>10}")
    for i, name in enumerate(["dead/severe", "severe/moderate",
                              "moderate/healthy"]):
        shift = thresholds[i] - midpoints[i]
        print(f"  {name:<16} {midpoints[i]:>10.4f}  "
              f"{thresholds[i]:>10.4f}  {shift:>+10.4f}")