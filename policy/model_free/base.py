from abc import ABC, abstractmethod

class BasePolicy:
    def __init__(self, **kwargs):
        pass

    def process(self, **kwargs):
        pass
    
    @abstractmethod
    def learn(self, batch, **kwargs):
        pass
    
    @abstractmethod
    def choose_action(self, **kwargs):
        pass

  