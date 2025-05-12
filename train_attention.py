import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from sklearn.model_selection import train_test_split
from models.models import load_and_preprocess_data, create_sequences
from models.attention_model import build_attention_gru_model

# Initialize wandb
wandb.init(
    project="INM706",
    entity="INM706",
    config={
        "model": "GRU + Attention",
        "sequence_length": 30,
        "batch_size": 32,
        "epochs": 10,
        "learning_rate": 0.001,
        "dataset": "prices-split-adjusted.csv"
    }
)
config = wandb.config

# Load and preprocess data
data_path = "archive/prices-split-adjusted.csv"
sentiment_path = "sentiment_scores.csv"
df = load_and_preprocess_data(data_path, sentiment_path)
X, y = create_sequences(df, sequence_length=config.sequence_length)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build and train model
model = build_attention_gru_model(input_shape=(X_train.shape[1], X_train.shape[2]))
history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=config.epochs,
    batch_size=config.batch_size,
    verbose=1,
    callbacks=[
        WandbMetricsLogger(),
        WandbModelCheckpoint(filepath="attention_model.h5", monitor="val_loss", save_best_only=True)
    ]
)

# Evaluate model on test set
loss, accuracy = model.evaluate(X_test, y_test)
wandb.log({"test_loss": loss, "test_accuracy": accuracy})
wandb.finish()
