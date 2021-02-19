import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

#여러 환경에서 돌아가야 하는 코드를 공유할 때
USE_CUDA= torch.cuda.is_available()
DEVICE=torch.device("cuda" if USE_CUDA else "cpu")

#학습 데이터 전체를 총 몇 번이나 볼 것인가
EPOCHS=30
BATCH_SIZE=64

transform= transforms.Compose([
    transforms.ToTensor()
])

#학습용 트레이닝셋, 성능 평가용 테스트셋
trainset = datasets.FashionMNIST(
    root    ='./.data/',
    train   =True,
    download=True,
    transform=transform
)

testset=datasets.FashionMNIST(
    root='./.data/',
    train=False,
    download=True,
    transform=transform
)

#매개변수에 앞서 불러온 데이터셋을 넣어주고 배치 크기를 지정해준다.
train_loader=torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=BATCH_SIZE,
    shuffle= True
)
test_loader=torch.utils.data.DataLoader(
    dataset=testset,
    batch_size=BATCH_SIZE,
    shuffle = True
)

#변수들이 들어가는 연산 선언
#픽셀값 784를 입력받아, 가중치를 행렬곱하고 편향을 더해 256개를 출력
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self,x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#GPU를 사용하지만 CPU로 보낼 수도 있음
model=Net().to(DEVICE)
#SGD 모델 최적화를 위한 확률적 경사하강법
optimizer= optim.SGD(model.parameters(),lr=0.01)

#학습에 들어가는 모든 연산
def train(moder, train_loader, optimizer):
    model.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target= data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output=model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

#성능 측정하기
def evaluate(modes, test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy=100*correct/len(test_loader.dataset)
    return test_loss, test_accuracy

for epoch in range(1, EPOCHS+1):
    train(model, train_loader, optimizer)
    test_loss, test_accuracy = evaluate(model,test_loader)

    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch, test_loss, test_accuracy))
#다음으로 패션 아이템을 분류하는 코드를 실습해보겠습니다. 패션 아이템 이미지를 인식해 레이블을 예측하는 기본적인 심층 인공 신경망을 만들어보겠습니다. 모델은 입력x와 레이블 y를 받아 학습한 다음, 새로운 x가 왔을 때 어떤 패션 아이템인지 예측할 것입니다/.