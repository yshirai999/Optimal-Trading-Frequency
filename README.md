# Optimal-Trading-Frequency

- It is well known that reinforcement learning agents trading daily struggle to identify buying signals in the presence of microstructure noise and regime switches

- Here it is assumed that a buy/sell signal is observed by the agent, who then decides the investment amount and the investment horizon until the next trading day

- Specifically, a reinforcement learning environment is defined for an asset that follows a GBM, but its drift and volatility are subject to regime switches

- The agent observes a signal (buy or sell) and chooses the frequency of trading

- The higher the frequency, the lower the amount invested at each trading date

- Specifically, at each trading date, $fdt/T$ dollars of the asset are bought if drift is positive, and they are sold if the drift is negative where:
    - $f$ is investment horizon, decided by the traded
    - $dt$ unit time
    - $T$ final time

- The resulting Markov Decision Process is implemented in a gym environment, and an optimal policy is learned using PPO.
