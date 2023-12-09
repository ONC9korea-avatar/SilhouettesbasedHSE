import numpy as np

# 사용자에게 데이터셋의 개수를 입력받습니다.
total_count = int(input("데이터셋의 총 개수를 입력하세요: "))

# train set과 test set의 비율을 설정합니다. (예: 24:1)
train_ratio = 9
test_ratio = 1
ratio_sum = train_ratio + test_ratio

# 각 세트에 할당될 개수를 계산합니다.
train_count = (total_count * train_ratio) // ratio_sum
test_count = total_count - train_count

# Index를 생성하고 섞습니다.
indices = np.arange(total_count)
np.random.shuffle(indices)

# Index를 train set과 test set으로 나눕니다.
train_idx = indices[:train_count]
test_idx = indices[train_count:]

# npz 파일로 저장합니다.
np.savez('./train_test_index.npz', train_idx=train_idx, test_idx=test_idx)

print(f"Train set 개수: {len(train_idx)}, Test set 개수: {len(test_idx)}")
print(f"train_test_index.npz 파일이 저장되었습니다.")