import torch

x=torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(x)

print("Size:",x.size())
print("Shape:",x.shape)
print("랭크(차원):",x.ndimension()) #랭크 확인

#랭크 늘리기
#텐서 속 원소의 수는 유지
x=torch.unsqueeze(x,0)
print(x)
print("Size:",x.size())
print("Shape:",x.shape)
print("랭크(차원):", x.ndimension())

#랭크 줄이기
#총 원소 수는 영향을 받지 않음
x=torch.squeeze(x)
print(x)
print("Size:",x.size())
print("Shape:",x.shape)
print("랭크(차원):",x.ndimension())

#view는 텐서의 모양을 바꿀 수 있음
#텐서의 원소 개수는 바꿀 수 없음->[2,4] 불가능
x=x.view(9)
print("Size:",x.size())
print("Shape:",x.shape)
print("랭크(차원):",x.ndimension())
