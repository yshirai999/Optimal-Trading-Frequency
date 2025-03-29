# Optimal-Trading-Frequency

- Consider an asset that follows a GBM, but its drift and volatility are subject to regime switches

- The agent's action determines the trading' frequency.

- The observation is the sign of the drift, which is interpreted as a generic buy or sell signal

- At each trading date, $fdt/T$ dollars of the asset are bought if drift is positive, and they are sold if the drift is negative

- The resulting Markov Decision Process is implemented in a gym environment, and an optimal policy is learned using PPO.

## Bugs to fix

- Check the time series of states: there should be no [0,1] and [1,0]

- Check what it means to predict[[0,1]]
