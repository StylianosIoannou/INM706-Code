#training script of the model, also where wandb logs the metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import wandb
from models.models import load_and_preprocess_data, create_sequences, SequenceDataset

# Define the simple baseline GRU model
class BaselineGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BaselineGRU, self).__init__()
        self.gru = nn.GRU(input_size=4, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # use the final time step's output
        out = self.dropout(out)
        out = self.fc(out)
        return self.sigmoid(out).squeeze(1)

# Initialize Weights & Biases
wandb.init(project="INM706", entity="INM706", config={
    "model": "BaselineGRU",
    "sequence_length": 30,
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 0.001
})
config = wandb.config

# Load and preprocess data
df = load_and_preprocess_data()
X, y = create_sequences(df, sequence_length=config.sequence_length)
dataset = SequenceDataset(X, y)

# Train/Val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=False)
val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BaselineGRU(input_size=5, hidden_size=50).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Training loop
for epoch in range(config.epochs):
    model.train()
    total_loss, correct = 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        correct += ((outputs > 0.5) == y_batch).sum().item()

    # Validation
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            val_outputs = model(X_val)
            loss = criterion(val_outputs, y_val)
            val_loss += loss.item() * X_val.size(0)
            val_correct += ((val_outputs > 0.5) == y_val).sum().item()

    train_acc = correct / len(train_set)
    val_acc = val_correct / len(val_set)

    # Log to WandB
    wandb.log({
        "epoch": epoch,
        "train_loss": total_loss / len(train_set),
        "train_accuracy": train_acc,
        "val_loss": val_loss / len(val_set),
        "val_accuracy": val_acc
    })

wandb.finish()
