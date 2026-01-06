import json
import os

notebook_path = r'c:\DLProject\notebooks\resnet_binary_classification_gradcam.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Transform Cell
target_code_1 = "img_size = 224 \ntransform = transforms.Compose([ # 이미지 전처리 파이프라인 구성\n    transforms.Resize((img_size, img_size)), # 이미지 크기를 224x224로 통일\n    transforms.ToTensor(), # 이미지를 파이토치 텐서 형식으로 변환\n    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 데이터 정규화 (채널별 평균 및 표준편차)\n]) \n"

replacement_code_1 = """img_size = 224 

# 학습용: 데이터 증강(Augmentation) 포함
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(p=0.5),      # 50% 확률로 좌우 반전
    transforms.RandomRotation(degrees=15),       # 최대 15도 회전
    transforms.ColorJitter(brightness=0.2),      # 밝기 조절
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 검증 및 테스트용: 원본 특징 유지를 위해 증강 제외
test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
"""

# Update Dataset Cell
target_code_2 = "# Dataset 및 DataLoader도 분할에 맞게 재정의\ntrain_dataset = CastingDataset(train_df, transform=transform)\nval_dataset = CastingDataset(val_df, transform=transform) \ntest_dataset = CastingDataset(test_df, transform=transform) \n"

replacement_code_2 = """# Dataset 및 DataLoader도 분할에 맞게 재정의
train_dataset = CastingDataset(train_df, transform=train_transform)
val_dataset = CastingDataset(val_df, transform=test_transform) 
test_dataset = CastingDataset(test_df, transform=test_transform) 
"""

def normalize_source(source):
    if isinstance(source, list):
        return "".join(source)
    return source

updated = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = normalize_source(cell['source'])
        
        if "img_size = 224" in source_str and "transforms.Compose" in source_str and "transform =" in source_str:
            cell['source'] = [line + "\n" for line in replacement_code_1.split("\n")][:-1]
            # Need to handle the last line newline carefully if split removes it
            # Actually, let's just assign individual lines
            cell['source'] = [line + "\n" for line in replacement_code_1.splitlines()]
            updated = True
        
        if "CastingDataset(train_df" in source_str and "transform=transform" in source_str:
            cell['source'] = [line + "\n" for line in replacement_code_2.splitlines()]
            updated = True

if updated:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook updated successfully.")
else:
    print("Could not find targets in notebook.")
