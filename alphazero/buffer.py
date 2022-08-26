import random
from dataclasses import dataclass
import torch


@dataclass
class Sample:
    state: list
    mtcs_policy: list
    player: int
    reward: int
    mask: list


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.records = []
        self.index = 0

    def add_record(self, record):
        """
        Parameters
        -----
        record: List[Sample]
        """
        if len(self.records) >= self.buffer_size:
            for sample in record:
                index = random.randrange(self.buffer_size)
                self.records[index] = sample
        else:
            self.records.extend(record)
        if len(self.records) > self.buffer_size:
            self.records = self.records[:-1]

    def get_minibatch(self, batch_size):
        """
        Parameters
        -----
        batch_size: int

        Returns
        -----
        states: torch.Tensor(batch_size, 2, 15, 15)
        masks: torch.Tensor(batch_size, 2, 15, 15)
        mtcs_policy: torch.Tensor(batch_size, 2 * 15 *15)
        reward: torch.Tensor(batch_size, 1)
        """
        batchs = random.sample(self.records, batch_size)
        states = torch.cat([torch.unsqueeze(torch.from_numpy(batch.state).clone(), 0) for batch in batchs], 0)
        masks = torch.cat([torch.unsqueeze(torch.from_numpy(batch.mask).clone(), 0) for batch in batchs], 0)
        mtcs_policy = torch.cat([torch.unsqueeze(torch.Tensor(batch.mtcs_policy).clone(), 0) for batch in batchs], 0)
        reward = torch.cat([torch.unsqueeze(torch.Tensor([batch.reward]).clone(), 0) for batch in batchs], 0)

        return states, masks, mtcs_policy, reward

    def __len__(self):
        return len(self.records)
