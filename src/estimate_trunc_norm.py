import numpy as np
import json
import scipy.stats as st
import scipy.optimize as opt
import matplotlib.pyplot as plt


# --- 1. Simulate data from a doubly‑truncated Normal -------------------------
np.random.seed(42)
true_mu, true_sigma = 111.0, 10.0
a, b = 80.0, 160.0  # truncation limits
n = 500  # sample size

with open("sample_data.json", "r") as f:
    data = json.load(f)

x = np.array(data["weights"])


# --- 2. Log‑likelihood (negative, for minimisation) --------------------------
def neg_ll(params, data, a, b):
    mu, log_sigma = params
    sigma = np.exp(log_sigma)  # enforce σ>0
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    Z = st.norm.cdf(beta) - st.norm.cdf(alpha)  # normalising constant
    ll = -np.sum(
        np.log(sigma)
        + np.log(Z)
        + 0.5 * ((data - mu) / sigma) ** 2
        + 0.5 * np.log(2 * np.pi)
    )
    return -ll  # we minimise −log L


# --- 3. Initial guesses: ignore truncation -----------------------------------
mu0, sigma0 = x.mean(), x.std()
result = opt.minimize(neg_ll, x0=[mu0, np.log(sigma0)], args=(x, a, b), method="BFGS")
mu_hat, sigma_hat = result.x[0], np.exp(result.x[1])

print(f"True parameters   : μ = {true_mu: .3f}, σ = {true_sigma: .3f}")
print(f"Estimated (MLE)   : μ̂ = {mu_hat: .3f}, σ̂ = {sigma_hat: .3f}")

# --- 4. Visual diagnostic -----------------------------------------------------
xs = np.linspace(a, b, 400)
alpha_hat = (a - mu_hat) / sigma_hat
beta_hat = (b - mu_hat) / sigma_hat
Z_hat = st.norm.cdf(beta_hat) - st.norm.cdf(alpha_hat)
pdf_hat = st.norm.pdf((xs - mu_hat) / sigma_hat) / (sigma_hat * Z_hat)

plt.figure()
plt.hist(x, bins=30, density=True, edgecolor="black", alpha=0.5)
plt.plot(xs, pdf_hat, linewidth=2)
plt.axvline(a, linestyle="--")
plt.axvline(b, linestyle="--")
plt.title("Histogram of simulated data with fitted truncated‑Normal pdf")
plt.xlabel("x")
plt.ylabel("Density")
plt.show()
