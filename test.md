# Assignment: Computational Finance

**Names:** Samuel Lavalaye, Wouter van der Heide, Nicolas Wolniewicz
**Student Numbers:** [ID 1], [ID 2], u2120794

---

## Question 1: Monte Carlo Pricing of an Asian Option

### 1.1 Methodology
To price the arithmetic Asian call option, we simulated the stock price path over 52 weeks using the Black-Scholes dynamics.

* **Payoff Function:** C_T = max(Average_S - K, 0)
    * Where **Average_S** is the arithmetic average of weekly prices.
* **Simulation:** We performed **R = 500,000** replications. Sample size was chosen to reduce the standard error sufficiently so that the 95% Confidence Interval (CI) width is **<= 0.15**.

### 1.2 Results
* **Estimated Option Price:** ~ 5.76
* **95% Confidence Interval:** [5.72, 5.80]
* **Interval Width:** ~ 0.08 

### 1.3 Discussion: Challenges of Path-Dependent Options
Pricing path-dependent options like the Asian option presents unique challenges compared to standard European options:

1.  **Dimensionality & Memory:** A European option is "path-independent," meaning its payoff depends only on the final price **S_T**. This allows us to simulate just one time step. An Asian option requires relying on the average of prices over time. This requires simulating the entire trajectory (52 steps),  increasing the dimensionality of the problem and the computational cost.
2.  **Analytically Complex:** The sum (or average) of log-normally distributed variables does not follow a log-normal distribution. In most cases, there is no simple closed-form solution for the Asian options, making Monte Carlo simulations or moment-matching approximations necessary.

---

## Question 2: Computation of Greeks for a Digital Option

**Option parameters:** Digital Call, S0 = 100, K = 105, T = 1.5, sigma = 20%, r = 2%.
**Payoff:** Pays 10 if S_T >= K, else 0.

### 2.1 Delta Estimates
*Definition: Sensitivity of price to change in underlying price S0.*

| Method | Estimate | Applicability & Issues |
| :--- | :--- | :--- |
| **Bump-and-Reprice** | **~ 0.16** | It is sensitive to the bump size *h* (bias-variance trade-off) and more computationally expensive. |
| **Pathwise Method** | **N/A** | **Not Applicable.** The payoff is a step function (discontinuous at K). This makes the pathwise differentiation impossible. |
| **Likelihood Ratio** | **~ 0.16** | **Applicable.** This method differentiates the probability density function rather than the payoff function. It handles discontinuous payoffs well but can have higher variance than pathwise methods. |

### 2.2 Vega Estimates
*Definition: Sensitivity of price to change in volatility (sigma).*

| Method | Estimate | Applicability & Issues |
| :--- | :--- | :--- |
| **Bump-and-Reprice** | **~ 0.18** | **Applicable.** Similar to Delta.  |
| **Pathwise Method** | **N/A** | **Not Applicable.** The payoff is a step function (discontinuous at K). This makes the pathwise differentiation impossible. |
| **Likelihood Ratio** | **~ 0.18** | **Applicable.** Provides an unbiased estimate by weighting the payoff with a "score" function derived from the density's sensitivity to sigma. |

---

## Question 3: Discrete-Time Hedging and Risk Management
**Option Parameters**: European Digital Call: S0 = 100, K = 105, T = 1.5.

**Payoff:** Pays 10 if S_T >= K, else 0.

**Assumptions**: We work in a Black-Scholes market with parameters: mu=9%, sigma=20%, r=2%, S_0 = 100, B_0 = 1. A financial instution sells 2000 Euopean digital call options. The institution wants to hedge the associated risk using a discrete-time delta hedging strategy on a weekly rebalancing grid. The position in the money market account is adjusted to ensure the portfolio is self financing. The initial capital is the premium received from selling the options.

### 3a Initial Positions

The initial digital call price can be calculated with the following formula, which has been derived with the First Fundamental Theorem of Asset Pricing:

$$
\begin{aligned}
    C_t &= 10  e^{-r(T-t)} N(d_2) \\
    d_2 &= \frac{\ln(S_t / K) + (r - \frac{1}{2}\sigma^2)(T-t)}{\sigma \sqrt{T-t}}.
\end{aligned}
$$

Plugging in the correct values and applying budget neutrality results in the following numbers.

| Metric | Symbol | Value |
| :--- | :---: | :--- |
| **Digital Call Price ($t=0$)** | $C_0$ | $4.0861$ |
| **Initial Stock Position** | $\phi_0$ | $309.8991$ |
| **Initial Bond Position** | $\psi_0$ | $-22,817.6162$ |


### 3b Simulation Results

After simulating this weekly delta-hedging strategy over 2000 scenarios, we get the following plot and statistics for Profit and Loss (PnL). 

| Statistic | Value (EUR) |
| :--- | :--- |
| **Mean PnL** | -51.82 |
| **Standard Deviation** | 2,827.29 |
| **5% Quantile** | -4,624.27 |
| **95% Quantile** | 4,228.20 |

A plot of the different quantiles of portfolio value over time has been added a seperate image to this assignment. 

### 3c Risk Premium
In our results we we see that the average PnL of our strategy is almost equal to zero. However, we also see that the standard deviation is quite large, and that the 5% quantile lies around -4500 euros. This means that there is a reasably large probability that we lose a large amount of money when using these strategies. This happens because we are following a weekly rebalancing scheme, instead of a continuous scheme. This exposes us to pin risk. To avoid this, we would like to charge a risk premium. 

We can calculate the Value at Risk (VaR). We look at the 95% VaR, which is a measure of the magnitude of the loss at the 5% quantile of our P&L distribtuion. From our simulation results, the 5% quantile is approximately -4624. This means that in 95% of market scenarios, the totall loss (hedging error) will not exceed 4624.

To protect against these 'normal' hedging errors, we propose charging this 95% VaR as an additional risk premium. This would mean that the customer covers the risk for 95% of market scenarios. We can calculate the premium per option.

$$
\text{Risk Premium} = \frac{VaR}{N_{options}} = \frac{4624.27}{2000} \approx \text{EUR } 2.31
$$

## Appendix: Python Code

```python
import numpy as np
import scipy.stats as stats

# Set seed for reproducibility
np.random.seed(42)

# ==========================================
# QUESTION 1: Asian Option Pricing
# ==========================================

# Imports
import numpy as np
from scipy.stats import norm

# Parameters
T = 1 # Payoff at meturity
S_0 = 100 # Current stock price
Sigma = 0.25 # Volatility or underlying assets returns
K = 102 # Strike price
r = 0.02 # Riskfree rate
n = 52 # Amount of weeks

# NR of MC replications
R = 500000

# Price option at t=0 (based upon closed-form formula)
d1 = (np.log(S_0 / K) + (r + 0.5 * Sigma**2) * T) / (Sigma * np.sqrt(T))
d2 = d1 - Sigma * np.sqrt(T)
price = S_0 * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)
print("Exact option price:", price)

# Price option at t=0 (based upon MC approximation)
Z =  np.random.normal(0, 1, size=(R, n))
increment = ((r - 0.5 * Sigma**2) * (T/n) + Sigma * np.sqrt(T/n) * Z)
log_S = np.log(S_0) + np.cumsum(increment, axis=1)
S_path = np.exp(log_S)
S_mean = S_path.mean(axis=1)
C_T   = np.maximum(S_mean - K, 0.0)

price_MC = np.exp(-r * T) * np.mean(C_T)
print("MC approximation option price:", price_MC)
std_CT = np.std(C_T, ddof=1)

price_MC_CI_left  = np.exp(-r * T) * (np.mean(C_T) - 1.96 * std_CT / np.sqrt(R))
price_MC_CI_right = np.exp(-r * T) * (np.mean(C_T) + 1.96 * std_CT / np.sqrt(R))
CI_width = price_MC_CI_right - price_MC_CI_left

print(f"95% CI for option price: ({price_MC_CI_left}, {price_MC_CI_right})")
print("Width of confidence interval: ", CI_width)

# ==========================================
# QUESTION 2: Digital Option Greeks
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

## Mean
mu = 0.09
## Standard Deviation
sigma = 0.2
## Initial Stock Price
s0 = 100
## Interest Rate
r = 0.02
## Time Window
T = 1.0
## Strike Price
K = 105
## Number of simulations
n = 1000000

# Estimate Delta Bump-Reprice
def delta_pump_reprice(n, s0, T, sigma, r, K, h):
    Wq_T = np.sqrt(T) * stats.norm.rvs(loc=0, scale=1, size=n)

    ST = s0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * Wq_T)

    payoff = 10 * (ST >= K)
    option_price = np.exp(-r * T) * np.mean( payoff)

    ST_bump = (s0 + h) * np.exp((r - 0.5 * sigma ** 2) * T + sigma * Wq_T)

    payoff_bump = 10 * (ST_bump >= K)
    option_price_bump = np.exp(-r * T) * np.mean(payoff_bump)

    return (option_price_bump - option_price) / h

# Pathwise method not applicable, the payoff is binary, non differentiable

# Estimate Delta Likelihood
def delta_lrm(n, s0, T, sigma, r, K):
    Wq_T = np.sqrt(T) * stats.norm.rvs(loc=0, scale=1, size=n)
    ST = s0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * Wq_T)
    payoff = 10 * (ST >= K)
    score_function = Wq_T / (s0 * sigma * T)

    return np.exp(-r * T) * np.mean(payoff * score_function)

# Estimate Vega Bump-Reprice
def vega_bump_reprice(n, s0, T, sigma, r, K, h):
    Wq_T = np.sqrt(T) * stats.norm.rvs(loc=0, scale=1, size=n)

    ST = s0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * Wq_T)

    payoff = 10 * (ST >= K)
    option_price = np.exp(-r * T) * np.mean( payoff)

    ST_bump = s0 * np.exp((r - 0.5 * (sigma + h) ** 2) * T + (sigma + h) * Wq_T)

    payoff_bump = 10 * (ST_bump >= K)
    option_price_bump = np.exp(-r * T) * np.mean(payoff_bump)

    return (option_price_bump - option_price) / h

# Estimate Vega Likelihood
def vega_lrm(n, s0, T, sigma, r, K):
    Wq_T = np.sqrt(T) * stats.norm.rvs(loc=0, scale=1, size=n)
    ST = s0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * Wq_T)
    
    score_function = -(1/sigma) - Wq_T + (Wq_T ** 2)/(sigma*T)

    payoff = 10 * (ST >= K)

    return np.exp(-r * T) * np.mean(payoff * score_function)

# Calculate Paramteres
delta_bump = delta_pump_reprice(n, s0, T, sigma, r, K, h=0.01)
delta_likelihood = delta_lrm(n, s0, T, sigma, r, K)
print(f'Delta Bump and Reprice: {delta_bump}')
print(f'Delta Likelihood Ratio: {delta_likelihood}')

vega_bump = vega_bump_reprice(n, s0, T, sigma, r, K, h=0.01)
vega_likelihood = vega_lrm(n, s0, T, sigma, r, K)
print(f'Vega Bump and Reprice: {vega_bump}')
print(f'Vega Likelihood Ratio: {vega_likelihood}')


# ==========================================
# QUESTION 3: Discrete-Time Hedging and Risk Management
# ==========================================

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class BlackScholes_market:
    """
    A class to represent a Black-Scholes market environment
    """

    def __init__(self, mu, r, sigma, S0, B0):
        self.mu = mu
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.B0 = B0
        
    
    def simulate_stock_price(self, time_interval, seed = None):
        """
        Simulate stock price using Geometric Brownian Motion
        
        :param time_interval: A discretized array of time

        returns 
        An array of simulated stock prices at each time step
        """
        if seed is not None:
            np.random.seed(seed)

        dt = np.diff(time_interval)
        W = np.random.standard_normal(size=len(dt))
        
        W = np.cumsum(np.sqrt(dt) * W) # Brownian motion
        W = np.concatenate(([0], W)) #W[0] = 0

        S_t = self.S0 * np.exp((self.mu - 0.5 * self.sigma**2) * time_interval + self.sigma * W)
        return S_t

class DigitalCallOption:
    """
    A class to represent a European Digital Call Option with payoff 
    a * 1_{S_T > K} at maturity T
    """

    def __init__(self, K, T, a=1.0):

        self.K = K
        self.T = T
        self.a = a

    
    def payoff(self, S_T):
        """
        Calculate the payoff of the digital call option at maturity
        
        :param S_T: Stock price at maturity T
        
        returns 
        The payoff of the digital call option
        """
        if S_T > self.K:
            return self.a
        else:
            return 0.0
    
    def price(self, S_t, r, sigma, t=0.0):
        """
        Calculate the price of the digital call option at time t using Black-Scholes formula
        The price has been calculated analytically using the first fundamental
        theorem of asset pricing (i.e. risk-neutral valuation).

        :param S_t: Stock price at time t
        :param r: interest rate
        :param sigma: volatility of the underlying asset
        :param t: Current time
        
        returns 
        The price of the digital call option at time t
        """
        d2 = (np.log(S_t / self.K) + (r - 0.5 * sigma**2) * (self.T - t)) / (sigma * np.sqrt(self.T - t))
        price = self.a * np.exp(-r * (self.T - t)) * norm.cdf(d2)
        return price
    
    def delta(self, S_t, r, sigma, t=0.0):
        """
        Calculate the Delta of the digital call option at time t

        :param S_t: Current stock price
        :param r: interest rate
        :param sigma: volatility of the underlying asset
        :param t: Current time
        """

        d2 = (np.log(S_t / self.K) + (r - 0.5 * sigma**2) * (self.T - t) )/ (sigma * np.sqrt(self.T - t))
        delta = self.a * np.exp(-r * (self.T - t)) * norm.pdf(d2) / (S_t * sigma * np.sqrt(self.T - t))
        return delta
    
def run_hedging_simulation(market, option, time_interval):
    option_amount = 2000
    
    #Simulate a stock price
    S = market.simulate_stock_price(time_interval)
    S0 = S[0]
    B0 = 1.0
    V_t_series = []


    C0 = option.price(S0, market.r, market.sigma, t=0)
    current_option_delta = option.delta(S0, market.r, market.sigma, t=0)

    option_premium = C0 * option_amount

    #Delta hedging strategy
    phi = option_amount*current_option_delta
    psi = (option_premium - S0 * phi )/B0

    wealth = phi * S0 + psi * B0
    V_t_series.append(wealth)

    for (i,t) in enumerate(time_interval[1::]):
        S_t = S[i]
        B_t = np.exp(market.r * t)

        #Calculate wealth before rebalancing
        wealth = phi * S_t + psi * B_t
        V_t_series.append(wealth)

        #Calculate new strategy (if not at maturity)
        if i < len(time_interval)-2:
            current_option_delta = option.delta(S_t, market.r, market.sigma, t=t)

            phi = option_amount * current_option_delta
            psi = (wealth - phi*S_t)/B_t


        else: #Final time, t=T
            payoff = option_amount*option.payoff(S_t)
            PnL = wealth - payoff #Profit and Loss

    return (V_t_series, PnL, S)
        
        

mu = 0.09 # Drift of underlying stock
sigma = 0.2 # Volatility of underlying stock
S0 = 100 # Initial stock price
B0 = 1 # Initial bond value
r = 0.02 #Interest rate on risk free bond
K = 105 # Strike price of digital call option
T = 1.5 # Maturity of digital call option

N_sims = 2000 # Number of simulations

option_amount = 2000 # Number of options to hedge in the portfolio

rebalancing_interval = 1/52 #Weekly rebalancing

n_rebalancing = int(T/rebalancing_interval) #How many times we rebalance for our hedge.

#Exercice a
# digital_option = DigitalCallOption(K=K, T=T, a=10)
# C0 = digital_option.price(S_t=S0, r=r, sigma=sigma, t=0.0)


# current_option_delta = digital_option.delta(S0, r, sigma, t=0)

# option_premium = C0 * option_amount

# #Delta hedging strategy
# phi = 2000*current_option_delta
# psi = (option_premium - S0 * phi )/B0

# wealth = psi * B0 + phi * S0

# print(f"Digital Call Option Price at t=0: {C0}")
# print(f'Initial option premium received: {option_premium}')
# print(f"Initial hedge position in stock: {phi}")
# print(f"Initial hedge position in bond: {psi}")
# print(f"Initial wealth of the hedging portfolio: {wealth}")


# #Exercise b

#Setup data structures to store results
wealth_simulated = np.zeros((N_sims, n_rebalancing+1))
PnL_simulated = np.zeros(N_sims)
time = np.linspace(0, T, n_rebalancing+1) # Time discretization

#Run simulations
market_model = BlackScholes_market(mu=mu, r=r, sigma=sigma, S0=S0, B0=B0)
digital_option = DigitalCallOption(K=K, T=T, a=10)


for i in range(N_sims):
    if i % 100 == 0:
        print(f"Running simulation {i+1}/{N_sims}")
    wealth, PnL, S =run_hedging_simulation(market_model, digital_option, time)

    wealth_simulated[i,:] = wealth
    PnL_simulated[i] = PnL

#Compute statistics
quantiles_5 = np.percentile(wealth_simulated, 5, axis=0)
quantiles_50 = np.percentile(wealth_simulated, 50, axis=0)
quantiles_95 = np.percentile(wealth_simulated, 95, axis=0)

PnL_mean = np.mean(PnL_simulated)
PnL_std = np.std(PnL_simulated)
PnL_5 = np.percentile(PnL_simulated, 5)
PnL_95 = np.percentile(PnL_simulated, 95)

#Make plots
plt.plot(time, quantiles_95, label='95% Quantile', color='green')
plt.plot(time, quantiles_50, label='Median (50%)', color='blue')
plt.plot(time, quantiles_5,  label='5% Quantile', color='red')


plt.xlabel('Time (years)')
plt.ylabel('Portfolio Value')
plt.title("Wealth process of weekly delta-hedged portfolio for Digital Call Option")
plt.legend()
plt.show()


plt.hist(PnL_simulated, bins=50, edgecolor='black')
plt.title('Histogram of PnL from Delta Hedging Digital Call Option')
plt.xlabel('PnL')
plt.ylabel('Frequency')
plt.show()

#Print results
print(f"Average PnL over {N_sims} simulations: {np.mean(PnL_simulated)}")
print(f"Standard Deviation of PnL over {N_sims} simulations: {np.std(PnL_simulated)}")
print(f"5% Quantile of PnL over {N_sims} simulations: {PnL_5}")
print(f"95% Quantile of PnL over {N_sims} simulations: {PnL_95}")


#Exercise c
VaR = -PnL_5 #The 5% quantile of the PnL distribution gives us the VaR at 95% confidence level
print(f"Value at Risk (VaR) at 95% confidence level: {VaR}")

risk_premium_per_option = VaR / option_amount
total_risk_premium = risk_premium_per_option * option_amount

print(f"Risk premium per option to cover 95% VaR: {risk_premium_per_option}")
