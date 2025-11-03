from typing import Iterable
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.app import ArrayRecord, ConfigRecord, Message


class FedProx(FedAvg):
    def __init__(self, mu: float = 0.01, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.mu=mu
        print(f"FedProx initialized, mu is = {self.mu}")
    def configure_train(self, server_round: int, 
                        arrays: ArrayRecord, config: ConfigRecord, 
                        grid: Grid)-> Iterable[Message]:
    
        config["mu"] = self.mu

        print(f"Round {server_round}: Configuring FedProx with mu={config['mu']}")
        return super().configure_train(server_round, arrays, config, grid)
