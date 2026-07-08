import torch
import matplotlib.pyplot as plt


def emp(beta):
    torch.random.manual_seed(42)
    temperature = torch.randn([30]) * 10
    exp_weight_avg = []
    for idx, data in enumerate(temperature, 1):
        if idx == 1:
            exp_weight_avg.append(data)
            continue
        new_tempt = beta * exp_weight_avg[idx - 2] + (1 - beta) * data
        exp_weight_avg.append(new_tempt)
    plt.plot(range(len(exp_weight_avg)), exp_weight_avg)
    plt.scatter(range(len(exp_weight_avg)), temperature)
    plt.show()


if __name__ == '__main__':
    emp(0)
    emp(0.2)
    emp(0.9)
