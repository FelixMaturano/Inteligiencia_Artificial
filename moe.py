import torch
import torch.nn as nn


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts

        assignments = torch.nonzero(gates > 0, as_tuple=False)
        self._batch_index = assignments[:, 0]
        self._expert_index = assignments[:, 1]
        self._part_sizes = (gates > 0).sum(dim=0).int().tolist()
        self._nonzero_gates = gates[self._batch_index, self._expert_index]

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index]
        return list(torch.split(inp_exp, self._part_sizes, dim=0))

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, dim=0)
        if multiply_by_gates:
            stitched = stitched * self._nonzero_gates.view(-1, 1)

        zeros = torch.zeros(
            self._gates.size(0),
            stitched.size(1),
            device=stitched.device,
            dtype=stitched.dtype,
        )
        combined = zeros.index_add(0, self._batch_index, stitched)
        return combined

    def expert_to_gates(self):
        return list(torch.split(self._nonzero_gates, self._part_sizes, dim=0))


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class MoE(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_experts, noisy_gating=True, k=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k

        self.experts = nn.ModuleList([
            MLP(self.input_size, self.output_size, self.hidden_size) for _ in range(self.num_experts)
        ])

        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)

        assert self.k <= self.num_experts

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0.0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean().pow(2) + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(dim=0)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k, self.num_experts), dim=1)
        top_k_gates = top_logits / (top_logits.sum(dim=1, keepdim=True) + 1e-6)

        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_indices, top_k_gates)
        load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(dim=0)
        aux_loss = self.cv_squared(importance) + self.cv_squared(load)
        aux_loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, aux_loss
