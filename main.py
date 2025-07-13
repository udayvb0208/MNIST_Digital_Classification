import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import squarify
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset
train_df = pd.read_csv(r"C:\Users\krama\Downloads\archive - 2025-05-21T235247.047\mnist_train.csv")
test_df = pd.read_csv(r"C:\Users\krama\Downloads\archive - 2025-05-21T235247.047\mnist_test.csv")

# EDA
print("\n📊 Dataset Overview")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
print("Class distribution:\n", train_df['label'].value_counts())
plt.figure(figsize=(8, 4))
sns.countplot(x=train_df['label'])
plt.title("Digit Distribution")
plt.show()
plt.imshow(train_df.iloc[0, 1:].values.reshape(28, 28), cmap='gray')
plt.title(f"Label: {train_df.iloc[0, 0]}")
plt.axis('off')
plt.show()
all_pixels = train_df.iloc[:, 1:].values.flatten()
plt.figure(figsize=(10, 5))
plt.hist(all_pixels, bins=50, color='skyblue', edgecolor='black')
plt.title("Histogram of All Pixel Intensities (MNIST Train Set)")
plt.xlabel("Pixel Intensity (0–255)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 8))
subset = train_df.iloc[:, 1:101]
corr = subset.corr()
sns.heatmap(corr, cmap="coolwarm", cbar=True)
plt.title("Heatmap of Pixel Correlations (First 100 Pixels)")
plt.show()
plt.figure(figsize=(6, 6))
label_counts = train_df['label'].value_counts().sort_index()
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab10.colors)
plt.title("Digit Distribution (Pie Chart)")
plt.axis('equal')
plt.show()
plt.figure(figsize=(8, 5))
squarify.plot(sizes=label_counts.values, label=label_counts.index, alpha=0.8, color=plt.cm.tab10.colors)
plt.title("Treemap of Digit Classes")
plt.axis('off')
plt.show()

# Prepare data
X_train_np = train_df.iloc[:, 1:].values / 255.0
y_train_np = train_df.iloc[:, 0].values
X_test_np = test_df.iloc[:, 1:].values / 255.0
y_test_np = test_df.iloc[:, 0].values

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.transform(X_test_np)

# Traditional ML models
models = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}
print("\n🔧 Training Traditional ML Models:")
for name, model in models.items():
    model.fit(X_train_scaled, y_train_np)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test_np, preds)
    results[name] = acc
    print(f"\n📈 {name} Evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test_np, preds, digits=4))

# Prepare for CNN
X_train_torch = torch.tensor(X_train_np.reshape(-1, 1, 28, 28), dtype=torch.float32)
y_train_torch = torch.tensor(y_train_np, dtype=torch.long)
X_test_torch = torch.tensor(X_test_np.reshape(-1, 1, 28, 28), dtype=torch.float32)
y_test_torch = torch.tensor(y_test_np, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=128)

# CNN Model Definition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize and train CNN
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n🧠 Training CNN with PyTorch:")
for epoch in range(50):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# CNN Evaluation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cnn_acc = accuracy_score(all_labels, all_preds)
results['CNN'] = cnn_acc
print(f"\n📈 CNN Evaluation:")
print(f"Accuracy: {cnn_acc:.4f}")
print(classification_report(all_labels, all_preds, digits=4))

# Bar chart for comparison
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), [v * 100 for v in results.values()], color='lightgreen')
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.ylim(80, 100)
plt.show()
