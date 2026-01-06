import json
import os

notebook_path = r'c:\DLProject\notebooks\resnet_binary_classification_gradcam.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# replacement_code_1 with Data Augmentation
replacement_code_1 = [
    "img_size = 224 \n",
    "\n",
    "# 학습용: 데이터 증강(Augmentation) 포함\n",
    "train_transform = transforms.Compose([\n",
    "    transforms.Resize((img_size, img_size)),\n",
    "    transforms.RandomHorizontalFlip(p=0.5),      # 50% 확률로 좌우 반전\n",
    "    transforms.RandomRotation(degrees=15),       # 최대 15도 회전\n",
    "    transforms.ColorJitter(brightness=0.2),      # 밝기 조절\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])\n",
    "])\n",
    "\n",
    "# 검증 및 테스트용: 원본 특징 유지를 위해 증강 제외\n",
    "test_transform = transforms.Compose([\n",
    "    transforms.Resize((img_size, img_size)),\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])\n",
    "])\n"
]

# replacement_code_2 with updated transform names
replacement_code_2 = [
    "# Dataset 및 DataLoader도 분할에 맞게 재정의\n",
    "train_dataset = CastingDataset(train_df, transform=train_transform)\n",
    "val_dataset = CastingDataset(val_df, transform=test_transform) \n",
    "test_dataset = CastingDataset(test_df, transform=test_transform) \n"
]

def normalize_source(source):
    if isinstance(source, list):
        return "".join(source)
    return source

updated_count = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = normalize_source(cell['source'])
        
        # Match the first target cell
        if "img_size = 224" in source_str and "transform = transforms.Compose" in source_str:
            cell['source'] = replacement_code_1
            updated_count += 1
            print("Found and updated transform definition cell.")
        
        # Match the second target cell
        if "# Dataset 및 DataLoader도 분할에 맞게 재정의" in source_str:
            cell['source'] = replacement_code_2
            updated_count += 1
            print("Found and updated dataset assignment cell.")

if updated_count >= 2:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook updated successfully.")
else:
    print(f"Could only find {updated_count} out of 2 target cells. Manual check required.")
