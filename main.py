import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.data import load_data
from src.data import SodicDataset
from src.model import BertFinetuner


def run():
    pl.seed_everything(42)
    train_data, valid_data, test_data = load_data("data/train.csv", "data/test.csv")
    
    train_set = SodicDataset(train_data)
    valid_set = SodicDataset(valid_data)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, collate_fn=train_set.collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=16, shuffle=False, num_workers=4, collate_fn=valid_set.collate_fn)

    model = BertFinetuner()
    trainer = pl.Trainer(
        default_root_dir="outputs/bert",
        max_epochs=2,
        gpus=1,
        deterministic=True
    )
    trainer.fit(model, train_loader, valid_loader)

    test_set = SodicDataset(test_data)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4, collate_fn=test_set.collate_fn)
    preds = trainer.predict(model, test_loader)
    preds = torch.cat(preds, dim=0).detach().tolist()

    test_df = pd.read_csv("data/test.csv")
    test_df["label"] = preds
    result_df = test_df[["id", "label"]]
    result_df.to_csv("data/submission.csv", index=False)


if __name__ == "__main__":
    run()
