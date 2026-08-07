import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from dataset import Time_seriesDataset
from model import Transformer_enc
from trainer import Trainer
from data_utils import prepare_datasets
# Main function entry, illustrating the overall workflow

def main():
    Observe_length = 8
    prediction_length = 52
    batch_size = 32
    epochs = 200
    focus_classes = [1]
    model_dir = 'models'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "data"
    train_data, val_data = prepare_datasets(data_dir=data_dir)

    train_dataset = Time_seriesDataset(
        train_data, Observe_length, prediction_length,
        augment_data=True, focus_classes=focus_classes
    )
    train_scaler = train_dataset.get_scaler()
    train_label_encoder = train_dataset.get_label_encoder()
    val_dataset = Time_seriesDataset(
        val_data, Observe_length, prediction_length,
        augment_data=False, focus_classes=focus_classes,
        scaler=train_scaler, label_encoder=train_label_encoder
    )
    print("train_dataset_num:",len(train_dataset))
    print("val_dataset_num:",len(val_dataset))

    train_targets = train_dataset.encoded_labels
    class_weights = train_dataset.class_weights
    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    os.makedirs(model_dir, exist_ok=True)

    model = Transformer_enc(
        input_dim=train_dataset.feature_dim,
        d_model=128,
        nhead=8,
        num_layers=4,
        num_classes=train_dataset.num_classes,
        prediction_length=prediction_length,
        dropout=0.2,
        focus_classes=focus_classes
    ).to(device)

    trainer = Trainer(
        model, device, model_dir=model_dir,
        class_weights=train_dataset.class_weights,
        initial_reg_weight=1.0, initial_cls_weight=1.0,
        uncertainty_weighting=True, focus_classes=focus_classes
    )
    metrics = trainer.train(train_loader, val_loader, epochs, lr=0.0001)
    return metrics

if __name__ == "__main__":
    metrics = main()
