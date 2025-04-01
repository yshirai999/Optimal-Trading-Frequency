# Optimal-Trading-Frequency

- A reinforcement learning environment is defined for an asset that follows a GBM, but its drift and volatility are subject to regime switches

- The agent observes a signal (buy or sell) and chooses the frequency of trading

- The higher the frequency, the lower the amount invested at each trading date

- Specifically, at each trading date, $fdt/T$ dollars of the asset are bought if drift is positive, and they are sold if the drift is negative where:
    - $f$ is investment horizon, decided by the traded
    - $dt$ unit time
    - $T$ final time

- The resulting Markov Decision Process is implemented in a gym environment, and an optimal policy is learned using PPO.

## Bugs to fix

- Check the time series of states: there should be no [0,1] and [1,0]

- Check what it means to predict[[0,1]]
