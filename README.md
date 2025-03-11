# Optimal-Trading-Frequency

- Consider an asset that follows a GBM, but its drift and volatility are subject to regime switches

- The agent's action determines the trading' frequency.

- The observation is the sign of the drift, which is interpreted as a generic buy or sell signal

- At each trading date, $fdt/T$ dollars of the asset are bought if drift is positive, and they are sold if the drift is negative

- The resulting Markov Decision Process is implemented in a gym environment, and an optimal policy is learned using PPO.