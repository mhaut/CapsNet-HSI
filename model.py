import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


def squash(inputs, axis=-1):
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
    return scale * inputs


class DenseCapsule(nn.Module):
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings=3):
        super(DenseCapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings

        shape = torch.Size((out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))
        self.weight = nn.Parameter(0.01 * torch.randn(shape, out=torch.cuda.FloatTensor(shape)))

    def forward(self, x):
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)
        x_hat_detached = x_hat.detach()
        b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps)).cuda()

        assert self.routings > 0, 'The \'routings\' should be > 0.'
        for i in range(self.routings):
            c = F.softmax(b, dim=1)

            if i == self.routings - 1:
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
            else:  # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)

        return torch.squeeze(outputs, dim=-2)


class PrimaryCapsule(nn.Module):
    def __init__(self, in_channels, out_channels, dim_caps, kernel_size, stride=1, padding=0):
        super(PrimaryCapsule, self).__init__()
        self.dim_caps = dim_caps
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        outputs = self.conv2d(x)
        outputs = outputs.view(x.size(0), -1, self.dim_caps)
        return squash(outputs)

class CapsuleNet(nn.Module):
    def __init__(self, input_size, classes, routings,outpatchdim):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings
        ksize=3
        indimcap = 8
        inplanes = 256

        self.conv1 = nn.Conv2d(input_size[0], inplanes, kernel_size=ksize, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(inplanes)

        self.primarycaps = PrimaryCapsule(inplanes, inplanes, indimcap, kernel_size=ksize, stride=1, padding=0)
        self.digitcaps = DenseCapsule(in_num_caps=32*outpatchdim*outpatchdim, in_dim_caps=indimcap,
                                    out_num_caps=classes, out_dim_caps=indimcap*2, routings=routings)
        self.decoder = nn.Sequential(
            nn.Linear(16*classes, 328),
            nn.Sigmoid(),
            nn.Linear(328, 192),
            nn.Sigmoid(),
            nn.Linear(192, input_size[0] * input_size[1] * input_size[2])#,
        )

    def forward(self, x, y=None):
        x = self.relu(self.bn(self.conv1(x)))
        x = self.primarycaps(x)
        x = self.digitcaps(x)
        length = x.norm(dim=-1)
        if y is None:  # during testing, no label given. create one-hot coding using `length`
            index = length.max(dim=1)[1]
            y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).cuda())
        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        return length, reconstruction.view(-1, *self.input_size)


def caps_loss(y_true, y_pred, x, x_recon, lam_recon):
    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean()
    L_recon = nn.MSELoss()(x_recon, x)
    return L_margin + lam_recon * L_recon
