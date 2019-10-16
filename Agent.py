from abc import abstractmethod, ABC


class Agent(ABC):
    @abstractmethod
    def update(self, train=True):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def choose_action(self, state, prev_reward, done, train=True):
        pass
