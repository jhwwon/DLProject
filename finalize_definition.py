import json

notebook_path = r'c:\DLProject\notebooks\resnet_binary_classification_gradcam.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define the CastingDataset class code
class_definition_lines = [
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
    "\n"
]

updated = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_list = cell['source'] if isinstance(cell['source'], list) else [cell['source']]
        source_str = "".join(source_list)
        
        # We find the cell that is failing (the one with "# Dataset 및 DataLoader도 분할에 맞게 재정의")
        if "# Dataset 및 DataLoader도 분할에 맞게 재정의" in source_str:
            # Check if class is NOT already at the beginning of this specific cell
            if "class CastingDataset(Dataset):" not in source_str:
                # Prepend the class definition to the beginning of this cell
                cell['source'] = class_definition_lines + source_list
                updated = True
                print("Successfully prepended CastingDataset definition to the dataset creation cell.")

if updated:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook saved with consolidated definition.")
else:
    print("Dataset creation cell not found or already has definition.")
