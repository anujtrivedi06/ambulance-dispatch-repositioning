"""Paired significance testing between two policies.

Policies are compared on the same replications, so the right tests are the paired ones.
Both a parametric and a rank-based test are reported: coverage differences are roughly
symmetric, but response-time differences are not, and a conclusion that only one of the
two supports does not deserve to be stated as a result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class PairedResult:
    metric: str
    mean_a: float
    mean_b: float
    mean_diff: float  # b - a
    ci_low: float
    ci_high: float
    t_p: float
    wilcoxon_p: float
    cohens_d: float
    n: int

    @property
    def significant(self) -> bool:
        """Both tests agree at the 5% level and the CI excludes zero."""
        return (
            self.t_p < 0.05
            and self.wilcoxon_p < 0.05
            and (self.ci_low > 0) == (self.ci_high > 0)
        )

    def line(self) -> str:
        stars = "*" if self.significant else " "
        return (
            f"{self.metric:<22}{self.mean_a:>9.3f}{self.mean_b:>9.3f}"
            f"{self.mean_diff:>+10.3f}  [{self.ci_low:>+7.3f},{self.ci_high:>+7.3f}]"
            f"{self.t_p:>10.2g}{self.wilcoxon_p:>10.2g}{self.cohens_d:>8.2f} {stars}"
        )


def paired_test(
    a: np.ndarray, b: np.ndarray, metric: str = "metric", alpha: float = 0.05
) -> PairedResult:
    """Compare paired samples `a` (reference) and `b` on one metric."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same length")
    diff = b - a
    n = len(diff)
    mean_diff = float(diff.mean())
    sd = float(diff.std(ddof=1))

    if sd == 0.0:
        # Zero variance in the differences means one of two opposite things. If the mean
        # difference is also zero the policies are identical; if it is not, every single
        # replication moved the same way by the same amount, which is the strongest
        # evidence available rather than the weakest. The t statistic is undefined here
        # (zero standard error), so report the deterministic case directly.
        if mean_diff == 0.0:
            return PairedResult(metric, float(a.mean()), float(b.mean()), 0.0,
                                0.0, 0.0, 1.0, 1.0, 0.0, n)
        return PairedResult(
            metric=metric,
            mean_a=float(a.mean()),
            mean_b=float(b.mean()),
            mean_diff=mean_diff,
            ci_low=mean_diff,
            ci_high=mean_diff,
            t_p=0.0,
            wilcoxon_p=float(stats.wilcoxon(b, a).pvalue),
            cohens_d=float("inf") * np.sign(mean_diff),
            n=n,
        )

    se = sd / np.sqrt(n)
    crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    t_p = float(stats.ttest_rel(b, a).pvalue)
    w_p = float(stats.wilcoxon(b, a).pvalue)

    return PairedResult(
        metric=metric,
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=mean_diff,
        ci_low=mean_diff - crit * se,
        ci_high=mean_diff + crit * se,
        t_p=t_p,
        wilcoxon_p=w_p,
        cohens_d=mean_diff / sd,  # paired d: effect size in SDs of the difference
        n=n,
    )


def comparison_table(results: list[PairedResult], label_a: str, label_b: str) -> str:
    head = (
        f"{'metric':<22}{label_a[:8]:>9}{label_b[:8]:>9}{'diff':>10}"
        f"{'95% CI':>20}{'t p':>10}{'wilcox p':>10}{'d':>8}"
    )
    lines = [head, "-" * len(head)]
    lines += [r.line() for r in results]
    lines.append("")
    lines.append(f"* = significant at 5% on both tests (n = {results[0].n} paired reps)")
    return "\n".join(lines)
