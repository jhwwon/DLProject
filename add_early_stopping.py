import json

notebook_path = r'c:\DLProject\notebooks\resnet_binary_classification_gradcam.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define the training code with Early Stopping and Checkpointing
training_code = [
    "# ResNet18 모델 불러오기 및 수정\n",
    "\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "model = models.resnet18(pretrained=True)\n",
    "num_ftrs = model.fc.in_features\n",
    "model.fc = nn.Linear(num_ftrs, 2)  # 이진 분류\n",
    "model = model.to(device)\n",
    "\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = optim.Adam(model.parameters(), lr=1e-4)\n",
    "\n",
    "epochs = 20 # 얼리스탑핑을 확인하기 위해 에폭을 충분히 늘림\n",
    "train_loss_list = []\n",
    "val_loss_list = []\n",
    "train_acc_list = []\n",
    "val_acc_list = []\n",
    "\n",
    "# 체크포인트 및 얼리스탑핑 변수 설정\n",
    "best_val_loss = float('inf')\n",
    "patience = 3\n",
    "counter = 0\n",
    "model_save_path = '../models/resnet18_best.pth'\n",
    "os.makedirs('../models', exist_ok=True)\n",
    "\n",
    "for epoch in range(epochs):\n",
    "    model.train()\n",
    "    running_loss = 0.0\n",
    "    correct = 0\n",
    "    total = 0\n",
    "    for images, labels in tqdm(train_loader):\n",
    "        images, labels = images.to(device), labels.to(device)\n",
    "        optimizer.zero_grad()\n",
    "        outputs = model(images)\n",
    "        loss = criterion(outputs, labels)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        running_loss += loss.item() * images.size(0)\n",
    "        _, predicted = torch.max(outputs, 1)\n",
    "        correct += (predicted == labels).sum().item()\n",
    "        total += labels.size(0)\n",
    "    \n",
    "    train_loss = running_loss / total\n",
    "    train_acc = correct / total\n",
    "    train_loss_list.append(train_loss)\n",
    "    train_acc_list.append(train_acc)\n",
    "    \n",
    "    # 검증\n",
    "    model.eval()\n",
    "    val_loss = 0.0\n",
    "    val_correct = 0\n",
    "    val_total = 0\n",
    "    with torch.no_grad():\n",
    "        for images, labels in val_loader:\n",
    "            images, labels = images.to(device), labels.to(device)\n",
    "            outputs = model(images)\n",
    "            loss = criterion(outputs, labels)\n",
    "            val_loss += loss.item() * images.size(0)\n",
    "            _, predicted = torch.max(outputs, 1)\n",
    "            val_correct += (predicted == labels).sum().item()\n",
    "            val_total += labels.size(0)\n",
    "    \n",
    "    val_loss = val_loss / val_total\n",
    "    val_acc = val_correct / val_total\n",
    "    val_loss_list.append(val_loss)\n",
    "    val_acc_list.append(val_acc)\n",
    "    \n",
    "    print(f\"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\")\n",
    "    \n",
    "    # 체크포인트: 최적 모델 저장\n",
    "    if val_loss < best_val_loss:\n",
    "        best_val_loss = val_loss\n",
    "        torch.save(model.state_dict(), model_save_path)\n",
    "        print(f\"=> Best 모델 저장 완료 (Val Loss: {val_loss:.4f})\")\n",
    "        counter = 0\n",
    "    else:\n",
    "        counter += 1\n",
    "        print(f\"=> Val Loss 상승 ({counter}/{patience})\")\n",
    "        if counter >= patience:\n",
    "            print(\"Early Stopping 실행: 학습을 종료합니다.\")\n",
    "            break\n"
]

updated = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if "model = models.resnet18(pretrained=True)" in source and "epochs =" in source:
            cell['source'] = training_code
            updated = True
            break

if updated:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("SUCCESS: Early Stopping and Checkpointing implemented.")
else:
    print("ERROR: Training cell not found.")
