from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:         
        self.logger.record('reward', self.training_env.unwrapped.get_attr('reward')[0])
        self.logger.record('terminated', self.training_env.unwrapped.get_attr('terminated')[0])
        self.logger.record('truncated', self.training_env.unwrapped.get_attr('truncated')[0])
        self.logger.record('step', self.training_env.unwrapped.get_attr('time')[0])
        
        return True