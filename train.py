#training script of the model, also where wandb logs the metrics
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from sklearn.model_selection import train_test_split
from models.models import (
    load_and_preprocess_data,
    create_sequences,
    build_gru_model,
    train_model,
    evaluate_model
)

# Initialize wandb
wandb.init(
    project="INM706",
    entity="INM706", 
    config={
        "model": "GRU",
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

# Build model
model = build_gru_model(input_shape=(X_train.shape[1], X_train.shape[2]))

# Train model and log metrics manually
history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=config.epochs,
    batch_size=config.batch_size,
    verbose=1,
    callbacks=[
        WandbMetricsLogger(),
        WandbModelCheckpoint(filepath="model.h5", monitor="val_loss", save_best_only=True)
    ]
)

# Evaluate on test set
loss, accuracy = evaluate_model(model, X_test, y_test)

# Log final test metrics
wandb.log({"test_loss": loss, "test_accuracy": accuracy})

# End wandb run
wandb.finish()
