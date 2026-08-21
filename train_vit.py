import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import numpy as np
import pickle
import wandb
import timm
import time
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

device = torch.device('cuda')
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0))/1024**3:.1f}GB")

with open('data/data_config.pkl', 'rb') as f:
    config = pickle.load(f)

train_df      = config['train_df']
val_df        = config['val_df']
CLASSES       = config['classes']
class_weights = config['class_weights']
IMAGE_SIZE    = config['image_size']
SEG_DIR       = 'data/segmented_images'

FILENAME_COL = 'Image Index'
LABEL_COL    = 'Finding Labels'

def build_multihot(df, classes):
    label_lists  = df[LABEL_COL].str.split('|')
    multihot     = np.zeros((len(df), len(classes)), dtype=np.float32)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for row_idx, labels in enumerate(label_lists):
        for lbl in labels:
            lbl = lbl.strip()
            if lbl in class_to_idx:
                multihot[row_idx, class_to_idx[lbl]] = 1.0
    return multihot

class NIHDataset(Dataset):
    def __init__(self, df, image_dir, classes, transform=None):
        self.df        = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.labels    = build_multihot(df, classes)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname = self.df.iloc[idx][FILENAME_COL]
        img   = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx])

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = NIHDataset(train_df, SEG_DIR, CLASSES, train_transform)
val_dataset   = NIHDataset(val_df,   SEG_DIR, CLASSES, val_transform)

weight_vec     = np.array([class_weights[c] for c in CLASSES])
sample_weights = (train_dataset.labels * weight_vec).max(axis=1)
sampler        = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler,
                          num_workers=8, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False,
                          num_workers=8, pin_memory=True)

print(f"Train: {len(train_loader)} | Val: {len(val_loader)}")

model     = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=len(CLASSES)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

wandb.login()
wandb.init(project='xray-classification', name='03-classification-ViT')

os.makedirs('checkpoints/ViT', exist_ok=True)
CHECKPOINT_PATH = 'checkpoints/ViT/latest.pt'
BEST_PATH       = 'checkpoints/ViT/best.pt'

NUM_EPOCHS           = 50
EARLY_STOP_PATIENCE  = 7
start_epoch          = 0
best_val_auc         = 0.0
epochs_since_improvement = 0

if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    start_epoch              = ckpt['epoch'] + 1
    best_val_auc             = ckpt['best_val_auc']
    epochs_since_improvement = ckpt.get('epochs_since_improvement', 0)
    print(f"Resumed from epoch {start_epoch}, best AUC: {best_val_auc:.4f}")
else:
    print("Starting fresh!")

def run_epoch(loader, training=True):
    model.train() if training else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc="Train" if training else "Val"):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    aucs = [roc_auc_score(all_labels[:, i], all_preds[:, i])
            for i in range(len(CLASSES)) if all_labels[:, i].sum() > 0]
    return total_loss / len(loader), np.mean(aucs)

for epoch in range(start_epoch, NUM_EPOCHS):
    t0 = time.time()
    train_loss, train_auc = run_epoch(train_loader, training=True)
    val_loss,   val_auc   = run_epoch(val_loader,   training=False)
    scheduler.step(val_auc)
    elapsed = time.time() - t0

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} AUC: {val_auc:.4f} | {elapsed:.0f}s")
    wandb.log({'epoch': epoch+1, 'train_loss': train_loss, 'train_auc': train_auc,
               'val_loss': val_loss, 'val_auc': val_auc})

    if val_auc > best_val_auc:
        best_val_auc             = val_auc
        epochs_since_improvement = 0
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'val_auc': val_auc}, BEST_PATH)
        print(f"  💾 Best saved (AUC: {val_auc:.4f})")
    else:
        epochs_since_improvement += 1

    torch.save({
        'epoch': epoch, 'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_val_auc': best_val_auc,
        'epochs_since_improvement': epochs_since_improvement
    }, CHECKPOINT_PATH)

    if epochs_since_improvement >= EARLY_STOP_PATIENCE:
        print("Early stopping!")
        break

wandb.log({'final_best_val_auc': best_val_auc})
print(f"Done! Best AUC: {best_val_auc:.4f}")
