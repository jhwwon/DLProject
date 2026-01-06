import json

notebook_path = r'c:\DLProject\notebooks\resnet_binary_classification_gradcam.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define the robust combined cell content
combined_source = [
    "class CastingDataset(Dataset): # 주조 제품 전용 데이터셋 클래스 정의\n",
    "    def __init__(self, df, transform=None): \n",
    "        self.df = df.reset_index(drop=True) \n",
    "        self.transform = transform \n",
    "    def __len__(self): \n",
    "        return len(self.df)\n",
    "    def __getitem__(self, idx):\n",
    "        img_path = self.df.loc[idx, 'img_path'] \n",
    "        label = self.df.loc[idx, 'label'] \n",
    "        image = Image.open(img_path).convert('RGB') # 이미지 파일 읽기\n",
    "        if self.transform: \n",
    "            image = self.transform(image) \n",
    "        return image, label \n",
    "\n",
    "# Dataset 및 DataLoader도 분할에 맞게 재정의\n",
    "train_dataset = CastingDataset(train_df, transform=train_transform)\n",
    "val_dataset = CastingDataset(val_df, transform=test_transform) \n",
    "test_dataset = CastingDataset(test_df, transform=test_transform) \n",
    "\n",
    "from torch.utils.data import DataLoader\n",
    "batch_size = 32  # 커널 충돌 방지를 위해 배치 사이즈를 32로 설정\n",
    "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)\n",
    "val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)\n",
    "test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)\n"
]

updated = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if "# Dataset 및 DataLoader도 분할에 맞게 재정의" in source:
            cell['source'] = combined_source
            updated = True
            break

if updated:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("SUCCESS: Combined class definition and loader creation in one cell.")
else:
    print("ERROR: Target cell not found.")
