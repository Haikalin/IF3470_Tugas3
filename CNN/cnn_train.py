import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import scipy.io as sio

#Konfigurasi Global Variabel
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 183
BATCH_SIZE = 2
LR = 1e-4
EPOCHS = 3
IMAGE_SIZE = 256
CHECKPOINT_INTERVAL = 300

# Dataset
class COCO10kDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None, target_transform=None):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.transform = transform
        self.target_transform = target_transform

        self.images = sorted([p for p in self.img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        self.mats   = sorted([p for p in self.ann_dir.iterdir() if p.suffix.lower() == ".mat"])

        if len(self.images) == 0 or len(self.mats) == 0:
            raise RuntimeError("Dataset kosong! Pastikan images/ dan annotations/ berisi data.")

        if len(self.images) != len(self.mats):
            print("[WARNING] Jumlah gambar != jumlah file .mat")

    def __len__(self):
        return len(self.images)

    def load_mat_mask(self, mat_path):
        m = sio.loadmat(mat_path)

        for key in ["S", "mask", "segmentation", "GTcls", "GTseg"]:
            if key in m:
                arr = m[key]
                arr = np.squeeze(arr)
                arr = arr.astype(np.int32)
                return arr

        raise RuntimeError(f"Tidak menemukan mask di file .mat: {mat_path}")

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mat_path = self.mats[idx]

        img = Image.open(img_path).convert("RGB")
        mask_np = self.load_mat_mask(mat_path)
        mask = Image.fromarray(mask_np)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)

        return img, mask


# Transforms
img_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.NEAREST),
    transforms.PILToTensor(),
    transforms.Lambda(lambda t: t.squeeze().long())
])


# Model (Pretrained)
print("\nLoading DeepLabV3-ResNet50 pretrained...")

model = deeplabv3_resnet50(
    weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
)

# Replace classification head → 183 classes
model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=LR)


# Training
def train(img_dir, ann_dir):

    dataset = COCO10kDataset(
        img_dir=img_dir,
        ann_dir=ann_dir,
        transform=img_transform,
        target_transform=mask_transform
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    total_batches = len(loader)

    print(f"\nTotal batches per epoch: {total_batches}")
    print("Start training...\n")

    global_batch = 0
    best_loss_last50 = float('inf')
    best_checkpoint_last50 = None

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(loader, total=len(loader), desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch_idx, (imgs, masks) in enumerate(loop, start=1):
            global_batch += 1

            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(imgs)["out"]
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            loop.set_postfix(loss=current_loss)

            # =====================================================
            # CHECKPOINT SETIAP 300 BATCH
            # =====================================================
            if global_batch % CHECKPOINT_INTERVAL == 0:
                ckpt_path = f"check_{epoch + 1}_{global_batch}_{current_loss:.4f}.pth"

                torch.save({
                    "epoch": epoch,
                    "batch": global_batch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "loss": current_loss
                }, ckpt_path)

                print(f"\n>>> CHECKPOINT DISIMPAN: {ckpt_path}\n")

            # =====================================================
            # BEST DI 50 BATCH TERAKHIR EPOCH 3
            # =====================================================
            is_last_epoch = (epoch == EPOCHS - 1)
            is_last_50_batches = (batch_idx > total_batches - 50)

            if is_last_epoch and is_last_50_batches:
                if current_loss < best_loss_last50:
                    best_loss_last50 = current_loss
                    
                    best_checkpoint_last50 = {
                        "epoch": epoch,
                        "batch": global_batch,
                        "batch_in_epoch": batch_idx,
                        "model_state": model.state_dict(),
                        "optim_state": optimizer.state_dict(),
                        "loss": current_loss
                    }
                    
                    print(f"\n🏆 NEW BEST in Last 50 Batches: Loss={current_loss:.4f} at batch {batch_idx}/{total_batches}\n")


    # =====================================================
    # SIMPAN MODEL TERBAIK DI 50 BATCH TERAKHIR
    # =====================================================
    if best_checkpoint_last50 is not None:
        best_path = f"best_last50_batch{best_checkpoint_last50['batch']}_loss{best_loss_last50:.4f}.pth"
        torch.save(best_checkpoint_last50, best_path)
        print(f"\n🎯 BEST MODEL (Last 50 Batches) DISIMPAN: {best_path}")
        print(f"   Loss: {best_loss_last50:.4f}")
        print(f"   Batch in Epoch 3: {best_checkpoint_last50['batch_in_epoch']}/{total_batches}\n")

    print("\nTraining selesai!")
    torch.save(model.state_dict(), "deeplab_resnet50_coco10k_final_3epoch.pth")
    print("Model final disimpan sebagai deeplab_resnet50_coco10k_final_3epoch.pth")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    train(
        img_dir="PCD3/images",
        ann_dir="PCD3/annotations"
    )
