import torch
from torch.optim import Optimizer

class FISTA(Optimizer):
    def __init__(self, params, lr=1e-2, gamma=0.1):
        defaults = dict(lr=lr, gamma=gamma)
        super(FISTA, self).__init__(params, defaults)

    def step(self, decay=1, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:  #p代表要更新的权重，即m=mask
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if 'alpha' not in state or decay:#在刚开始迭代,state为空或者decay为True时
                    state['alpha'] = torch.ones_like(p.data)
                    state['data'] = p.data
                    y = p.data
                else:
                    alpha = state['alpha']
                    data = state['data']
                    state['alpha'] = (1 + (1 + 4 * alpha**2).sqrt()) / 2   #系数alpha
                    y = p.data + ((alpha - 1) / state['alpha']) * (p.data - data)  #中间变量y
                    state['data'] = p.data

                mom = y - group['lr'] * grad
                p.data = self._prox(mom, group['lr'] * group['gamma'])  #更新后的权重m

                # no-negative,保持权重非负。
                p.data = torch.max(p.data, torch.zeros_like(p.data))

        return loss

    def _prox(self, x, gamma):
        y = torch.max(torch.abs(x) - gamma, torch.zeros_like(x))

        return torch.sign(x) * y   #torch.sign是符号函数，大于0的元素对应1，小于0的元素对应-1，0还是0
