import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.RNN):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(-1) #x.view(x.size(0), -1)


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class SimpleBody(nn.Module):
    def __init__(self, num_channels):
        super(SimpleBody, self).__init__()
        self.out_feats = 32
        self.fc1 = nn.Linear(num_channels, self.out_feats) 
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x
    


class DQNBase(BaseNetwork):

    def __init__(self, num_channels, num_actions):
        super(DQNBase, self).__init__()
        self.gru_size = 128
        self.body = SimpleBody(num_channels)
        self.gru_net = nn.GRU(input_size = self.body.out_feats, hidden_size= self.gru_size, num_layers = 2,batch_first=True )
        self.lf = nn.Linear(self.gru_size, num_actions)

    def forward(self, states):

        #format outp for batch first rnn
        feats = self.body(states)
        x, _ = self.gru_net(feats)
        x = self.lf(x)
        x = x[:,-1,:]# select the current time of action estimation
        return x


class QNetwork(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()

        if not shared:
            self.gru_net = DQNBase(num_channels, num_actions)

        if not dueling_net:
            self.head = nn.Sequential(
                nn.Linear(num_actions, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_actions))
        # else:
        #     self.a_head = nn.Sequential(
        #         nn.Linear(7 * 7 * 64, 512),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(512, num_actions))
        #     self.v_head = nn.Sequential(
        #         nn.Linear(7 * 7 * 64, 512),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(512, 1))

        self.shared = shared
        self.dueling_net = dueling_net

    def forward(self, _states):
        if not self.shared:
            states = self.gru_net(_states)

        if not self.dueling_net:
            return self.head(states)
        # else:
        #     a = self.a_head(states)
        #     v = self.v_head(states)
        #     return v + a - a.mean(1, keepdim=True)


class TwinnedQNetwork(BaseNetwork):
    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()
        self.Q1 = QNetwork(num_channels, num_actions, shared, dueling_net)
        self.Q2 = QNetwork(num_channels, num_actions, shared, dueling_net)

    def forward(self, _states):
        q1 = self.Q1(_states)
        q2 = self.Q2(_states)
        return q1, q2


class CateoricalPolicy(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False): # 3, 25 separately
        super().__init__()
        if not shared:
            self.gru_net = DQNBase(num_channels, num_actions)

        self.head = nn.Sequential(
            nn.Linear(num_actions, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_actions))

        self.shared = shared

    def act(self, _states):
        if not self.shared:
            states = self.gru_net(_states)

        action_logits = self.head(states)
        greedy_actions = torch.argmax(
            action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, _states):
        if not self.shared:
            states = self.gru_net(_states)

        action_probs = F.softmax(self.head(states), dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs


