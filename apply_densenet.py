import json
import os

notebook_path = r'c:\DLProject\notebooks\resnet_binary_classification_gradcam.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# replacement_code_comparison with DenseNet121
replacement_code_comparison = [
    "import gc # 필요한 라이브러리 임포트\n",
    "import torch.nn as nn # 신경망 구성 요소(레이어, 손실 함수 등)를 제공하는 모듈\n",
    "import torch.optim as optim # 모델 최적화 알고리즘(가중치 수정) 제공 모듈\n",
    "from torch import nn, optim\n",
    "from copy import deepcopy \n",
    "\n",
    "# 비교할 모델 리스트\n",
    "model_names = ['resnet18', 'densenet121'] \n",
    "model_dict = { \n",
    "    'resnet18': models.resnet18, \n",
    "    'densenet121': models.densenet121 \n",
    "} \n",
    "\n",
    "results = {} \n",
    "\n",
    "for name in model_names: \n",
    "    print(f'\\n===== {name.upper()} 학습 시작 =====')\n",
    "    \n",
    "    model = model_dict[name](weights='DEFAULT') \n",
    "    \n",
    "    if name == 'resnet18': \n",
    "        model.fc = nn.Sequential( \n",
    "            nn.Dropout(0.5), \n",
    "            nn.Linear(model.fc.in_features, 2) \n",
    "        ) \n",
    "    elif name == 'densenet121': \n",
    "        num_ftrs = model.classifier.in_features\n",
    "        model.classifier = nn.Sequential(\n",
    "            nn.Dropout(0.5),\n",
    "            nn.Linear(num_ftrs, 2)\n",
    "        )\n",
    "    \n",
    "    model = model.to(device)\n",
    "\n",
    "    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4) \n",
    "    criterion = nn.CrossEntropyLoss() \n",
    "\n",
    "    for epoch in range(3):\n",
    "        model.train() \n",
    "        train_loss, correct, total = 0, 0, 0 \n",
    "        for images, labels in train_loader: \n",
    "            images, labels = images.to(device), labels.to(device) \n",
    "            optimizer.zero_grad() \n",
    "            outputs = model(images) \n",
    "            loss = criterion(outputs, labels) \n",
    "            loss.backward() \n",
    "            optimizer.step() \n",
    "            train_loss += loss.item() * images.size(0) \n",
    "            _, predicted = outputs.max(1) \n",
    "            correct += predicted.eq(labels).sum().item() \n",
    "            total += labels.size(0) \n",
    "        train_acc = correct / total \n",
    "        print(f'Epoch {epoch+1}/3 | Train Acc: {train_acc:.4f}')\n",
    "\n",
    "    model.eval() \n",
    "    correct, total = 0, 0 \n",
    "    with torch.no_grad(): \n",
    "        for images, labels in val_loader: \n",
    "            images, labels = images.to(device), labels.to(device) \n",
    "            outputs = model(images) \n",
    "            _, predicted = outputs.max(1) \n",
    "            correct += predicted.eq(labels).sum().item() \n",
    "            total += labels.size(0) \n",
    "    val_acc = correct / total \n",
    "    print(f'{name.upper()} 검증 정확도: {val_acc:.4f}') \n",
    "    results[name] = val_acc \n",
    "    \n",
    "    del model \n",
    "    torch.cuda.empty_cache() \n",
    "    gc.collect() \n",
    "\n",
    "import pandas as pd \n",
    "result_df = pd.DataFrame(list(results.items()), columns=['Model', 'Val Accuracy']) \n",
    "display(result_df.sort_values('Val Accuracy', ascending=False)) "
]

def normalize_source(source):
    if isinstance(source, list):
        return "".join(source)
    return source

updated = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = normalize_source(cell['source'])
        if "model_names = ['resnet18'" in source_str and "mobilenet_v3_small" in source_str:
            cell['source'] = replacement_code_comparison
            updated = True
            print("Found and updated model comparison cell.")

if updated:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook updated successfully.")
else:
    print("Could not find the model comparison cell.")
