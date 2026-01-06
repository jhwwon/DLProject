import json

notebook_path = r'c:\DLProject\notebooks\resnet_binary_classification_gradcam.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define the CastingDataset class code
class_definition = [
    "\n",
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
    "        return image, label \n"
]

updated = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        # Insert into the transform cell (id 1202d585) or where img_size is defined
        if "img_size = 224" in source and "train_transform =" in source:
            # Check if class is already there (to avoid duplicates, though view_file says it's missing)
            if "class CastingDataset" not in source:
                # Add to the end of the cell
                if isinstance(cell['source'], list):
                    cell['source'].extend(class_definition)
                else:
                    cell['source'] += "".join(class_definition)
                updated = True
                print("Successfully added CastingDataset class definition to the transform cell.")
                break

if updated:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook saved with CastingDataset definition.")
else:
    print("Target cell for class definition not found.")
