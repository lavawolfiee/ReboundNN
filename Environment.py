from abc import abstractmethod, ABC


class Environment(ABC):
    states_num = 0
    actions_num = 0

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def available_actions(self, state):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self):
        pass
