from torchvision import datasets, transforms, utils
from torch.utils import data

import matplotlib.pyplot as plt
import numpy as np
#이미지를 텐서로 바꿔주는 코드
transform = transforms.Compose({
    transforms.ToTensor()
})
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
#데이터 개수, 반복마다 16갸씩 읽어준다
batch_size=16

#매개변수에 앞서 불러온 데이터셋을 넣어주고 배치 크기를 지정해준다.
train_loader = data.DataLoader(
    dataset=trainset,
    batch_size=batch_size
)
test_loader = data.DataLoader(
    dataset=testset,
    batch_size=batch_size
)
#next함수를 통해 배치 1개를 가져온다.
dataiter = iter(train_loader)
images, labels = next(dataiter)

#여러 이미지를 모아 하나의 이미지로 만든다.
img=utils.make_grid(images, padding=0)
npimg=img.numpy()
plt.figure(figsize=(10, 7))
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.show()

print(labels)
#이름 대신에 숫자 번호로 레이블, 딕셔너리를 만들어둠
CLASSES = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}
#영문 텍스트 출력
for label in labels:
    index = label.item()
    print(CLASSES[index])

#개별 이미지 꺼내기, 이미지 확인
idx=1
item_img=images[idx]
item_npimg=item_img.squeeze().numpy()
plt.title(CLASSES[labels[idx].item()])
plt.imshow(item_npimg, cmap='gray')
plt.show()