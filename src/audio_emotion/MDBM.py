

"""

다중 모달 학습과 딥 볼츠만 머신에 대한 더 구체적인 설명은 다음과 같습니다.

다중 모달 학습은 여러 유형의 데이터를 동시에 처리하여 서로 다른 모달(모드) 간의 관계를 파악하고 학습하는 기술입니다. 예를 들어, 이미지와 텍스트, 오디오와 같은 다양한 데이터가 동시에 사용되어 상호 보완적인 정보를 제공합니다.

딥 볼츠만 머신(DBM)은 깊은 비지도학습 모델로, 데이터의 복잡한 구조와 패턴을 찾아내는 데 효과적입니다. DBM은 여러 층의 은닉 유닛을 사용하여 데이터의 고차원적인 표현을 학습하며, 많은 양의 레이블되지 않은 데이터에서도 유용한 특징을 추출할 수 있습니다.

이 논문에서는 다중 모달 학습을 위한 딥 볼츠만 머신을 제안합니다. 이 모델은 각 모달에 대한 별도의 DBM을 사용하고, 상위 은닉층에서 다른 모달의 은닉층과 결합하여 공통의 표현을 학습합니다. 이 과정을 통해, 모델은 서로 다른 모달 간의 상관 관계를 파악하고, 통합된 특징을 사용하여 각 데이터 유형에 대한 예측 성능을 개선할 수 있습니다.

실험 결과로, 다중 모달 학습을 적용한 딥 볼츠만 머신은 단일 모달 학습 방법보다 더 나은 성능을 보였으며, 이미지와 텍스트 데이터를 함께 사용하여 더 정확한 분류와 예측이 가능함을 확인하였습니다. 이러한 연구 결과는 다중 모달 학습과 딥 볼츠만 머신을 활용하여 복잡한 문제를 해결하는 데 도움이 될 것입니다.






위 코드에서는 세 가지 데이터셋(MNIST, CIFAR-10, FashionMNIST)에 대해 각각 독립적인 DBM 모델을 학습시킵니다. 이렇게 학습된 모델은 각 데이터셋에 대해 특징을 추출할 수 있습니다.

다중 모달 학습의 경우, 모델이 다양한 데이터 유형을 동시에 처리하고 공동 특징을 학습하도록 구현해야 합니다. 이를 위해서는 서로 다른 모달 간의 상호 작용 및 공동 표현을 학습하는 방법을 추가해야 합니다. 이 작업은 본 논문에서 설명한 방법을 참고하여 추가적인 코드를 작성해야 합니다.


"""





### 필요한 라이브러리를 가져옵니다.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST

### 데이터셋을 불러오고 전처리를 수행합니다.
batch_size = 64

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_mnist = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = MNIST(root='./data', train=True, transform=transform_mnist, download=True)
cifar10 = CIFAR10(root='./data', train=True, transform=transform, download=True)
fashion_mnist = FashionMNIST(root='./data', train=True, transform=transform_mnist, download=True)

mnist_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)
cifar10_loader = DataLoader(cifar10, batch_size=batch_size, shuffle=True)
fashion_mnist_loader = DataLoader(fashion_mnist, batch_size=batch_size, shuffle=True)

### RBM 클래스를 정의합니다.
class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden, k=1, input_shape=None):
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
        self.input_shape = input_shape

        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.1)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def sample_h_given_v(self, v):
        probability = torch.sigmoid(F.linear(v, self.W.T, self.h_bias))
        return torch.bernoulli(probability)

    def sample_v_given_h(self, h):
        probability = torch.sigmoid(F.linear(h, self.W, self.v_bias))
        return torch.bernoulli(probability)



    def sample_from_probability(self, probability):
        return torch.bernoulli(probability)

    def contrastive_divergence(self, v, lr):
        # Positive phase
        h_prob = self.sample_h_given_v(v)
        positive_phase = torch.matmul(v.t(), h_prob)

        # Negative phase
        v_sample = self.sample_v_given_h(h_prob)
        h_sample = self.sample_h_given_v(v_sample)
        negative_phase = torch.matmul(v_sample.t(), h_sample)

        # Update weights and biases
        self.W.data += lr * (positive_phase - negative_phase) / v.size(0)
        self.v_bias.data += lr * torch.mean(v - v_sample, dim=0)
        self.h_bias.data += lr * torch.mean(h_prob - h_sample, dim=0)

        # Calculate reconstruction error
        error = torch.mean(torch.sum((v - v_sample) ** 2, dim=1))

        return error


    def free_energy(self, v):
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        visible_term = (v * self.v_bias).sum(1)
        return -hidden_term - visible_term

### 딥 볼츠만 머신(DBM) 모델을 정의합니다.
class DeepBoltzmannMachine(nn.Module):
    def __init__(self, layers, input_shapes=None):
        super(DeepBoltzmannMachine, self).__init__()
        self.layers = layers
        self.input_shapes = input_shapes or [None] * (len(layers) - 1)

        self.rbms = nn.ModuleList()
        for i in range(len(layers) - 1):
            input_shape = self.input_shapes[i] if i < len(self.input_shapes) else None
            self.rbms.append(RBM(layers[i], layers[i + 1], input_shape=input_shape))




    def forward(self, x):
        h = x
        for rbm in self.rbms:
            h = rbm.sample_h_given_v(h)
        return h

    def train_dbm(self, data_loader, input_shape, epochs=10, lr=0.01):
        for i, rbm in enumerate(self.rbms):
            print(f'Training layer {i+1} RBM:')
            for epoch in range(epochs):
                epoch_loss = 0
                if i == 0:
                    for batch_idx, (data, _) in enumerate(data_loader):
                        data = data.view(-1, np.prod(input_shape))
                        loss = rbm.contrastive_divergence(data, lr=lr)
                        epoch_loss += loss.item()
                else:  
                    for batch_idx, data in enumerate(data_loader):
                        data = data[0].view(-1, np.prod(input_shape))
                        loss = rbm.contrastive_divergence(data, lr=lr)
                        epoch_loss += loss.item()

                epoch_loss /= len(data_loader)
                print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}')
                
            if i < len(self.rbms) - 1:
                # Pretrain the next layer with samples from the current layer
                with torch.no_grad():
                    new_data = []
                    for data, _ in data_loader:
                        data = data.view(-1, np.prod(input_shape))
                        new_data.append(rbm.sample_h_given_v(data))
                    new_data = torch.cat(new_data, 0)
                    new_dataset = torch.utils.data.TensorDataset(new_data)
                    data_loader = DataLoader(new_dataset, batch_size=2, shuffle=True)




### 각 데이터셋에 대해 DBM 모델을 생성하고 학습시킵니다.
# MNIST
mnist_dbm = DeepBoltzmannMachine([1024, 500, 200], [(32, 32)])
mnist_dbm.train_dbm(mnist_loader, (32, 32), epochs=10, lr=0.01)

# CIFAR-10
cifar10_dbm = DeepBoltzmannMachine([32*32*3, 500, 200], [(3, 32, 32)])
cifar10_dbm.train_dbm(cifar10_loader, (3, 32, 32), epochs=10, lr=0.01)

# FashionMNIST
fashion_mnist_dbm = DeepBoltzmannMachine([32*32, 500, 200], [(32, 32)])
fashion_mnist_dbm.train_dbm(fashion_mnist_loader, (32, 32), epochs=10, lr=0.01)





### 먼저, 시각화를 위한 필요한 라이브러리를 가져옵니다.
import matplotlib.pyplot as plt
import numpy as np

### 샘플을 생성하고 시각화하는 함수를 정의합니다.
def generate_and_visualize_samples(dbm, input_shape, num_samples=10):
    # Initialize random input
    input_data = torch.bernoulli(torch.rand(num_samples, input_shape))

    # Generate samples
    with torch.no_grad():
        for rbm in dbm.rbms:
            input_data = rbm.sample_h_given_v(input_data)

        # Generate samples from the last layer
        generated_samples = dbm.rbms[-1].sample_v_given_h(input_data)

    # Rescale samples from [-1, 1] to [0, 1]
    generated_samples = (generated_samples + 1) / 2

    # Visualize samples
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i, ax in enumerate(axes):
        sample = generated_samples[i].view(*input_shape).cpu().numpy()
        if len(input_shape) == 2:
            ax.imshow(sample, cmap='gray')
        else:
            ax.imshow(np.transpose(sample, (1, 2, 0)))
        ax.axis('off')
    plt.show()

### 각 데이터셋에 대해 생성된 샘플을 시각화합니다.
# MNIST
print("Generated samples for MNIST:")
generate_and_visualize_samples(mnist_dbm, (32, 32))

# CIFAR-10
print("Generated samples for CIFAR-10:")
generate_and_visualize_samples(cifar10_dbm, (3, 32, 32))

# FashionMNIST
print("Generated samples for FashionMNIST:")
generate_and_visualize_samples(fashion_mnist_dbm, (32, 32))
















