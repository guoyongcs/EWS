import torch
import torch.nn as nn
from torch.nn import functional as F
import utils
import random

class Controller(nn.Module):
    def __init__(self, n_paths, n_channels, device, controller_type='LSTM', controller_hid=None,
                 controller_temperature=None, controller_tanh_constant=None, controller_op_tanh_reduce=None):
        super(Controller, self).__init__()
        assert len(n_channels) == len(n_paths), 'inconsistent number of elements in n_channels and n_paths'
        self.max_path = max(n_paths)
        self.max_channel = 0
        for i_channel in n_channels:
            for j_channel in i_channel:
                if j_channel > self.max_channel:
                    self.max_channel = j_channel
        self.max_embedding_length = self.max_path * (1 + self.max_channel)
        self.n_channels = n_channels
        self.n_paths = n_paths
        self.device = device
        self.controller_type = controller_type

        self.controller_hid = controller_hid
        self.attention_hid = self.controller_hid
        self.temperature = controller_temperature
        self.tanh_constant = controller_tanh_constant
        self.op_tanh_reduce = controller_op_tanh_reduce

        # Embedding of paths and channels
        self.path_channel_hidden = nn.ModuleList()
        for i in range(len(n_channels)):
            self.path_channel_hidden.append(
                nn.Embedding(self.max_embedding_length, self.controller_hid)
            )
        self.emb_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
        self.hid_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
        self.w_path = nn.Linear(self.controller_hid, self.max_path)
        self.w_channel = nn.Linear(self.controller_hid, self.max_channel)
        self.lstm = nn.LSTMCell(self.controller_hid, self.controller_hid)
        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)
        self.static_inputs = utils.keydefaultdict(self._get_default_hidden)
        self.tanh = nn.Tanh()
        self.selected_paths, self.selected_channels = [], []
        self.query_index = torch.LongTensor(range(0, max(self.max_path, self.max_channel))).to(device)
        self.eps = 1e-6

    def _get_default_hidden(self, key):
        return utils.get_variable(
            torch.zeros(key, self.controller_hid), self.device, requires_grad=False)

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (utils.get_variable(zeros, self.device, requires_grad=False),
                utils.get_variable(zeros.clone(), self.device, requires_grad=False))

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            if torch.rand(1).item() < 0.3:
                param.data.uniform_(-init_range, init_range)
        self.w_path.bias.data.fill_(0)
        self.w_channel.bias.data.fill_(0)

    def forward(self, sub_n_paths, sub_n_channels):
        log_p, entropy = 0, 0
        self.selected_paths, self.selected_channels = [], []
        path_action, channel_action = None, None
        inputs = self.static_inputs[1]  # batch_size x hidden_dim
        hidden = self.static_init_hidden[1]
        for i in range(len(self.n_channels)):
            # select paths
            hx, cx = self.lstm(inputs, hidden)
            logits = self.w_path(hx).index_select(1, self.query_index[0:self.n_paths[i]])
            if self.temperature is not None:
                logits /= self.temperature
            if self.tanh_constant is not None:
                op_tanh = self.tanh_constant / self.op_tanh_reduce
                logits = op_tanh * self.tanh(logits)
            # logits = logits * 0
            probs = F.softmax(logits, dim=-1) + self.eps
            log_probs = torch.log(probs)
            action = probs.multinomial(num_samples=sub_n_paths[i])
            path_action = torch.sort(action, dim=1).values
            self.selected_paths.append(path_action.squeeze(0).tolist())
            selected_log_p = log_probs.gather(1, action)
            selected_probs = probs.gather(1, action)
            log_p += selected_log_p.sum(1)[0]
            entropy += -(selected_log_p * selected_probs).sum()
            action = utils.get_variable(action, self.device, requires_grad=False)
            inputs = self.path_channel_hidden[i](action).mean(dim=1)
            hidden = (hx, cx)

            # select channels
            temp_selected_channel = []
            for j in path_action.squeeze(0).tolist():
                hx, cx = self.lstm(inputs, hidden)
                logits = self.w_channel(hx).index_select(1, self.query_index[0:self.n_channels[i][j]])
                if self.temperature is not None:
                    logits /= self.temperature
                if self.tanh_constant is not None:
                    op_tanh = self.tanh_constant / self.op_tanh_reduce
                    logits = op_tanh * self.tanh(logits)
                # logits = logits * 0
                probs = F.softmax(logits, dim=-1) + self.eps
                log_probs = torch.log(probs)
                action = probs.multinomial(num_samples=sub_n_channels[i][j])
                channel_action = torch.sort(action, dim=1).values
                temp_selected_channel.append(channel_action.squeeze(0).tolist())
                selected_log_p = log_probs.gather(1, action)
                selected_probs = probs.gather(1, action)
                log_p += selected_log_p.sum(1)[0]
                entropy += -(selected_log_p * selected_probs).sum()
                action = utils.get_variable(action, self.device, requires_grad=False)
                inputs = self.path_channel_hidden[i](self.max_path + j * (self.max_channel) + action).mean(dim=1)
                hidden = (hx, cx)
            self.selected_channels.append(temp_selected_channel)
        return self.selected_paths, self.selected_channels, log_p, entropy
