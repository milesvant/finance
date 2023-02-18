import numpy as np
from scipy import stats


class Underlying:
    def __init__(self, name, price, vol):
        """
        Object which represents underlying asset of an option.

        Parameters:
        name (str): Name of underlying asset.
        price (float): Current price of underlying asset.
        vol (float): Annualized volatility of underlying asset, represented as a decimal (e.g. 0.3 for 30% vol).
        """
        self.name = name
        self.price = price
        self.vol = vol


    def simulate_gbm(self, drift, steps, years, iterations):
        """
        Simulates price histories for asset using Geometric Brownian Motion (GBM) model.

        Parameters:
        drift (float): Annualized mean of GBM process.
        steps (int): Number of steps per iteration.
        years (float): Number of years, divided into `steps`, to simulate.
        iterations (int): Number of distinct price histories to simulate.

        Returns:
        np.ndarray: Price histories in shape (`steps`, `iterations`).
        """
        dt = years / steps
        returns = np.exp(
            (drift - self.vol ** 2 / 2) * dt
            + self.vol * np.random.normal(0, np.sqrt(dt), size=(steps, iterations))
        )
        return self.price * np.cumprod(returns, axis=0)


class Option:
    def __init__(self, underlying, k, t, r, option_type):
        """
        Object which represents and prices an option.

        Parameters:
        underlying (Underlying): Underlying asset of option.
        k (float): Strike price of option.
        t (float): Time to expiration in years.
        r (float): Risk-free rate.
        option_type (str): "call" or "put".
        """
        self.underlying = underlying
        self.k = k
        self.t = t
        self.r = r
        if option_type != "call" and option_type != "put":
            raise ValueError("Option must be put or call")
        self.option_type = option_type


    def intrinsic_value(self, prices):
        """
        Calculates non-discounted intrinsic value(s) of option.

        Parameters:
        prices (np.ndarray): Array of prices at expiration.

        Returns:
        np.ndarray: Instrinsic values in ndarray with same shape as `prices`.
        """
        if self.option_type == "call":
            return np.maximum(np.zeros_like(prices), prices - self.k)
        else:
            return np.maximum(np.zeros_like(prices), self.k - prices)


    def mc_price(self, granularity, iterations):
        """
        Prices option using Monte Carlo methods.

        Parameters:
        granularity (int): Number of timesteps per iteration to simulate.
        iterations (int): Number of Monte Carlo trials to use.
        """
        gbm = self.underlying.simulate_gbm(self.r, granularity, self.t, iterations)
        prices_at_expiry = self.intrinsic_value(gbm[-1, :])
        discounted = prices_at_expiry * np.exp(-self.r * self.t)
        return discounted.mean()


    def bs_price(self):
        """
        Prices option using the Black-Scholes formulas.
        """
        price = self.underlying.price
        vol = self.underlying.vol

        d1 = (np.log(price / self.k) + self.t * (self.r + vol ** 2 / 2)) / (vol * np.sqrt(self.t))
        d1_cdf = stats.norm.cdf(d1)
        d2 = d1 - vol * np.sqrt(self.t)
        d2_cdf = stats.norm.cdf(d2)
        discounted_strike = self.k * np.exp(-self.r * self.t)
        call_price = d1_cdf * price - d2_cdf * discounted_strike

        if self.option_type == "call":
            return call_price
        else:
            return discounted_strike - price + call_price


    def bs_greeks(self):
        """
        Calculates option greeks using Black-Scholes model.

        Returns:
        dict: Maps from greek name string (e.g. "gamma") to float value.
        """
        price = self.underlying.price
        vol = self.underlying.vol

        d1 = (np.log(price / self.k) + self.t * (self.r + vol ** 2 / 2)) / (vol * np.sqrt(self.t))
        d1_cdf = stats.norm.cdf(d1)
        d1_pdf = stats.norm.pdf(d1)
        d2 = d1 - vol * np.sqrt(self.t)
        d2_cdf = stats.norm.cdf(d2)
        neg_d2_cdf = stats.norm.cdf(-d2)
        discounted_strike = self.k * np.exp(-self.r * self.t)

        gamma = stats.norm.pdf(d1) / (price * vol * np.sqrt(self.t))
        vega = price * stats.norm.pdf(d1) * np.sqrt(self.t)

        if self.option_type == "call":
            delta = d1_cdf
            theta = - (price * d1_pdf * vol) / (2 * np.sqrt(self.t)) - self.r * discounted_strike * d2_cdf
            rho = discounted_strike * d2_cdf * self.t
        else:
            delta = d1_cdf - 1
            theta = - (price * d1_pdf * vol) / (2 * np.sqrt(self.t)) + self.r * discounted_strike * neg_d2_cdf
            rho = - discounted_strike * neg_d2_cdf * self.t

        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho,
        }
