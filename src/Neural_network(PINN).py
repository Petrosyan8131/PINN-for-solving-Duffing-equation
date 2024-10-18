import torch
import torch.nn as nn

# from Activation_functions.Activation_cos import Cos
from numerical_methods.Runge_kut_scipy import Approx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from numerical_methods.Runge_kut_approx import Runge_Kut

from torch.utils.data import Dataset, DataLoader
from Activation_functions.Activation_sin_cos import Sin, Cos

from tqdm import tqdm
# import optuna
# from numerical_methods.Runge_kut_approx import Runge_Kut
# from torch.utils.tensorboard import SummaryWriter

# a = Runge_Kut()
# b = a.method()

# set the parameters of the equation
gamma = 1.3
delta = 3
alpha = 0.001
beta = 0.0001
omega = torch.pi*1.25

# set the numerical approximation
a = Approx((gamma, delta, alpha, beta, omega))
b = a.solve()

num_data = torch.from_numpy(b.y[0, 1:]).unsqueeze(1)

matplotlib.rcParams['figure.figsize'] = (10.0, 7.0)

epohs = 6000
dots = 500

loss_all = np.zeros(epohs)
loss_all_num = np.zeros(epohs)
lambd = 1
# Используем доступные графические процессоры
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Задаем начальные данные для апроксимации (задание сетки точек)
t = (torch.linspace(0, 10.4, dots).unsqueeze(1)).to(device)
t.requires_grad = True
t_in = t[1:]
t_bc = t[0]

f_true = torch.zeros(dots-1).to(device)
f_true.requires_grad = True
g_true = torch.tensor([1., 0.]).to(device)
g_true.requires_grad = True

# Архитектура нейросети
class Neural(nn.Module):
    def __init__(self, hid_n=20):
        super(Neural, self).__init__()
        self.layers_stack = nn.Sequential(
        nn.Linear(1, hid_n),
        Sin(),
        nn.Linear(hid_n, hid_n), #1
        Sin(),
        nn.Linear(hid_n, hid_n), #2
        Sin(),
        nn.Linear(hid_n, hid_n), #3
        Sin(),
        nn.Linear(hid_n, 1),
    )

    def forward(self, x):
        return self.layers_stack(x)

# Создание модели PINN и её обучение
PINN = Neural().to(device)

metric_data = nn.MSELoss()
# writer = SummaryWriter()
optimizer = torch.optim.Adam(PINN.parameters()) #, lr=0.0001
criterion = nn.NLLLoss()


def pdeloss(t, epoh):
    out = PINN(t_in).to(device)
    f_in = pde(out, t_in)

    f_bc = PINN(t_bc).to(device)
    dxdt = torch.autograd.grad(f_bc, t_bc, torch.ones_like(t_bc), create_graph=True, retain_graph=True)[0]
    f_bc = torch.cat([f_bc, dxdt], 0)
    loss_pde = metric_data(f_in, f_true)
    loss_bc = metric_data(f_bc, g_true)
    # loss_tens = torch.pow(f_bc-g_true, 2)
    # loss_bc = loss_tens[0] + loss_tens[1]

    loss_num = torch.max(torch.abs(f_in - num_data))
    
    loss = loss_pde + loss_bc*lambd
    loss_all[epoh] = loss
    loss_all_num[epoh] = loss_num
    
    return loss

def pde(out, t):
        dxdt = torch.autograd.grad(out, t, torch.ones_like(t), create_graph=True, retain_graph=True)[0]
        d2xdt2 = torch.autograd.grad(dxdt, t, torch.ones_like(t), create_graph=True, retain_graph=True)[0]
        
        fxt = d2xdt2 + delta*dxdt + alpha*out + beta*out**3 - gamma*torch.cos(omega*t)
        return fxt

def train():
    pbar = tqdm(range(epohs), desc='Training Progress')
    for step in pbar:
        def closure():
            optimizer.zero_grad()
            loss = pdeloss(t, step)
            loss.backward()
            return loss

        optimizer.step(closure)

        if step % 2 == 0:
            current_loss = closure().item()
            pbar.set_description("Lambda: %.4f | Step: %d | Loss: %.7f" %
                                (lambd, step, current_loss))
    pbar.clear()

train()
# writer.close()

torch.save(PINN.state_dict(), r'./weights/weights_PINN_Duffing_equation.pth')
# filescolab.download("weights_PINN_harm_oscil_3exp.pth")

# initialize draw functions
def draw_approx(net, t):
    fs = 12
    # x = net.forward(t)
    x = PINN(t).to(device)
    z = torch.autograd.grad(x, t, torch.ones_like(t), create_graph=True, retain_graph=True)[0]
    plt.plot(t.detach().numpy(), x.detach().numpy(), '-', label=r"approx func")
    plt.plot(t.detach().numpy(), z.detach().numpy(), '-', label=r"first derivative")
    plt.plot(t.detach().numpy(), b.y[0, :], '-', label=r"numerical method")
    # plt.plot(b[2], b[0], '-', label=r"numerical_approx")
    plt.legend(fontsize=fs)
    plt.title('Approxing Duffing equation')
    plt.tight_layout()
    plt.savefig(r"./figs/approx_Duffing.png")
    plt.show()

def draw_history(net, t):
    fs = 12
    plt.plot(loss_all, label=r'Total Loss')
    plt.plot(loss_all_num, label=r'Loss With Numeral')

    ax=plt.gca()
    ax.set_yscale('log')
    plt.grid(which='major', linestyle='-')
    plt.grid(which='minor', linestyle='--')
    plt.xlim(0, epohs)
    plt.ylim(1e-5, 1e3)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    ax.tick_params(axis='both',direction='in')

    plt.legend(fontsize=fs)
    plt.xlabel('Iteration count', fontsize=fs)
    plt.ylabel('Loss', fontsize=fs)
    plt.title('Loss while training')
    plt.tight_layout()
    plt.savefig(r"./figs/history_Duffing.png")
    plt.show()

draw_approx(PINN, t)
draw_history(PINN, t)
