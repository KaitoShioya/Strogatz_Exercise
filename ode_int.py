from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


class NumIntegral:
    def __init__(self) -> None:
        self.algorithm_dict = {
            "euler_method": self.euler_method,
            "modified_euler_method": self.modified_euler_method,
            "runge_kutta_method": self.runge_kutta_method,
        }

    def initialize(
        self,
        x_0: np.ndarray,
        t_0: float,
        x_dot: Callable[[np.ndarray], np.ndarray],
        step: float,
        algorithm: str = "euler_method",
        custom_algorithm: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.t_0 = t_0
        self.x_0 = x_0
        self.x_dot = x_dot
        self.update_algorithm = algorithm
        if self.update_algorithm not in self.algorithm_dict.keys():
            if custom_algorithm is None:
                raise ValueError("algorighm not found and custom algorighm is None.")
            else:
                self.algorithm_dict[algorithm] = custom_algorithm
        self.step = step
        self.x = x_0
        self.t = t_0
        self._historical_t = [t_0]
        self._historical_x = [x_0]

    def run_integral(self, last_t: float, progress_bar: bool = True) -> None:
        t_range = [int(self.t_0 / self.step), int(last_t / self.step)]
        if progress_bar:
            for _ in tqdm(
                range(*t_range), desc="Numerical Integration Running Progress"
            ):
                self.update(self.x)
        else:
            for _ in range(*t_range):
                self.update(self.x)

    def update(self, x: np.ndarray) -> None:
        self.t += self.step
        self.x = self.x + self.algorithm_dict[self.update_algorithm](x)
        # 状態を記録
        self._historical_t.append(self.t)
        self._historical_x.append(self.x)

    def euler_method(self, x: np.ndarray) -> np.ndarray:
        return self.x_dot(x) * self.step

    def modified_euler_method(self, x: np.ndarray) -> np.ndarray:
        x_tilde = x + (self.x_dot(x) * self.step)
        return (self.x_dot(x) + self.x_dot(x_tilde)) * 0.5 * self.step

    def runge_kutta_method(self, x: np.ndarray) -> np.ndarray:
        k_1 = self.x_dot(x) * self.step
        k_2 = self.x_dot(x + (k_1 * 0.5)) * self.step
        k_3 = self.x_dot(x + (k_2 * 0.5)) * self.step
        k_4 = self.x_dot(x + k_3) * self.step
        return (k_1 + (2 * k_2) + (2 * k_3) + k_4) / 6

    @property
    def historical_t(self) -> np.ndarray:
        return np.array(self._historical_t)

    @property
    def historical_x(self) -> np.ndarray:
        return np.array(self._historical_x)

    @property
    def transition_record(self) -> pd.DataFrame:
        columns = ["x" + str(i) for i in range(1, self.historical_x.shape[-1] + 1)]
        return pd.DataFrame(self.historical_x, index=self.historical_t, columns=columns)