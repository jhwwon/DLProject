import json
import os

notebook_path = r'c:\DLProject\notebooks\resnet_binary_classification_gradcam.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The missing DataLoader lines with smaller batch size
replacement_code_loaders = [
    "# Dataset 및 DataLoader도 분할에 맞게 재정의\n",
    "train_dataset = CastingDataset(train_df, transform=train_transform)\n",
    "val_dataset = CastingDataset(val_df, transform=test_transform) \n",
    "test_dataset = CastingDataset(test_df, transform=test_transform) \n",
    "\n",
    "from torch.utils.data import DataLoader\n",
    "batch_size = 32  # 커널 충돌 방지를 위해 128에서 32로 하향 조정\n",
    "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)\n",
    "val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)\n",
    "test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)\n"
]

def normalize_source(source):
    if isinstance(source, list):
        return "".join(source)
    return source

updated = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = normalize_source(cell['source'])
        # Look for the cell that should contain datasets
        if "CastingDataset(train_df" in source_str and "train_dataset =" in source_str:
            cell['source'] = replacement_code_loaders
            updated = True
            print("Found and fixed DataLoader cell with batch_size=32.")

if updated:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook updated successfully.")
else:
    print("Could not find the target cell to fix.")
