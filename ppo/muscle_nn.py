"""
Standalone muscle network module for hierarchical control.

Pure PyTorch implementation with no Ray dependencies.
Compatible with both Python training (ppo_hierarchical.py) and C++ simulation (Environment.cpp).
"""

import torch
import torch.nn as nn
import numpy as np


def weights_init(m):
    """Xavier initialization for linear layers."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()


class MuscleNN(nn.Module):
    """
    Neural network for muscle activation control.

    Maps desired torques and muscle forces to muscle activations.
    Supports both standard and cascading (hierarchical) modes.
    """

    def __init__(self, num_total_muscle_related_dofs, num_dofs, num_muscles, is_cpu=False, is_cascaded=False):
        super(MuscleNN, self).__init__()

        self.num_total_muscle_related_dofs = num_total_muscle_related_dofs
        self.num_dofs = num_dofs  # Exclude Joint Root dof
        self.num_muscles = num_muscles
        self.isCuda = False
        self.isCascaded = is_cascaded

        num_h1 = 256
        num_h2 = 256
        num_h3 = 256

        self.fc = nn.Sequential(
            nn.Linear(num_total_muscle_related_dofs+num_dofs +
                      (num_muscles + 1 if self.isCascaded else 0), num_h1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h1, num_h2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h2, num_h3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h3, num_muscles),
        )

        # Normalization
        self.std_muscle_tau = torch.ones(
            self.num_total_muscle_related_dofs) * 200
        self.std_tau = torch.ones(self.num_dofs) * 200

        if torch.cuda.is_available() and not is_cpu:
            self.isCuda = True
            self.std_tau = self.std_tau.cuda()
            self.std_muscle_tau = self.std_muscle_tau.cuda()
            self.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.fc.apply(weights_init)

    def forward_with_prev_out_wo_relu(self, muscle_tau, tau, prev_out, weight=1.0):
        muscle_tau = muscle_tau / self.std_muscle_tau
        tau = tau / self.std_tau
        if type(prev_out) == np.ndarray:
            with torch.no_grad():
                prev_out = torch.FloatTensor(prev_out)
                out = prev_out + weight * \
                    self.fc.forward(
                        torch.cat([0.5 * prev_out, weight, muscle_tau, tau], dim=-1))
                return out
        else:
            out = prev_out + weight * \
                self.fc.forward(
                    torch.cat([0.5 * prev_out, weight, muscle_tau, tau], dim=-1))
            return out

    def forward_wo_relu(self, muscle_tau, tau):
        muscle_tau = muscle_tau / self.std_muscle_tau
        tau = tau / self.std_tau
        out = self.fc.forward(torch.cat([muscle_tau, tau], dim=-1))
        return out

    def forward(self, muscle_tau, tau):
        return torch.relu(torch.tanh(self.forward_wo_relu(muscle_tau, tau)))

    def forward_with_prev_out(self, muscle_tau, tau, prev_out, weight=1.0):
        return torch.relu(torch.tanh(self.forward_with_prev_out_wo_relu(muscle_tau, tau, prev_out, weight)))

    def unnormalized_no_grad_forward(self, muscle_tau, tau, prev_out=None, out_np=False, weight=None):
        with torch.no_grad():
            if type(self.std_muscle_tau) == torch.Tensor and type(muscle_tau) != torch.Tensor:
                if self.isCuda:
                    muscle_tau = torch.FloatTensor(muscle_tau).cuda()
                else:
                    muscle_tau = torch.FloatTensor(muscle_tau)

            if type(self.std_tau) == torch.Tensor and type(tau) != torch.Tensor:
                if self.isCuda:
                    tau = torch.FloatTensor(tau).cuda()
                else:
                    tau = torch.FloatTensor(tau)

            if type(weight) != type(None):
                if self.isCuda:
                    weight = torch.FloatTensor([weight]).cuda()
                else:
                    weight = torch.FloatTensor([weight])

            muscle_tau = muscle_tau / self.std_muscle_tau
            tau = tau / self.std_tau

            if type(prev_out) == type(None) and type(weight) == type(None):
                out = self.fc.forward(torch.cat([muscle_tau, tau], dim=-1))
            else:
                if self.isCuda:
                    prev_out = torch.FloatTensor(prev_out).cuda()
                else:
                    prev_out = torch.FloatTensor(prev_out)

                if type(weight) == type(None):
                    print('Weight Error')
                    exit(-1)
                out = self.fc.forward(
                    torch.cat([0.5 * prev_out, weight, muscle_tau, tau], dim=-1))

            if out_np:
                out = out.cpu().numpy()

            return out

    def forward_filter(self, unnormalized_activation):
        return torch.relu(torch.tanh(torch.FloatTensor(unnormalized_activation))).cpu().numpy()

    def load(self, path):
        print('load muscle nn {}'.format(path))
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(
                path, map_location=torch.device('cpu')))

    def save(self, path):
        print('save muscle nn {}'.format(path))
        torch.save(self.state_dict(), path)

    def get_activation(self, muscle_tau, tau):
        act = self.forward(torch.FloatTensor(muscle_tau.reshape(1, -1)).to(self.device),
                           torch.FloatTensor(tau.reshape(1, -1)).to(self.device))
        return act.cpu().detach().numpy()[0]

    def to(self, *args, **kwargs):
        """Override to() to update self.device"""
        self = super().to(*args, **kwargs)
        # Extract device from args or kwargs
        if args and isinstance(args[0], (torch.device, str)):
            self.device = torch.device(args[0])
        elif 'device' in kwargs:
            self.device = torch.device(kwargs['device'])
        else:
            # Infer device from parameters
            self.device = next(self.parameters()).device
        return self


def generating_muscle_nn(num_total_muscle_related_dofs, num_dofs, num_muscles, is_cpu=False, is_cascaded=False):
    """
    Factory function for C++ pybind11 compatibility.

    Args:
        num_total_muscle_related_dofs: Number of muscle-related DOFs
        num_dofs: Number of actuator DOFs (excluding root)
        num_muscles: Number of muscles
        is_cpu: Force CPU execution (default: False, uses GPU if available)
        is_cascaded: Enable cascading mode for hierarchical control (default: False)

    Returns:
        MuscleNN instance
    """
    return MuscleNN(num_total_muscle_related_dofs, num_dofs, num_muscles, is_cpu, is_cascaded)
