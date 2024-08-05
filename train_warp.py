from warp import WarpTrainer
from config_parse import ConfigParser
from imdb_dataset_warp import IMDBDataset
from torch.utils.data import DataLoader
import os


def main():
    
    config = ConfigParser().config
    train_dataset, _ = IMDBDataset(config=config).get_data()
    train_loader = DataLoader(train_dataset, batch_size=config["warp_model_parameters"]["batch_size"], shuffle=True, num_workers=15)
    trainer = WarpTrainer(config=config, dataloader=train_loader)
    trainer.train()
    trainer.save(out_name="alligned_llm")
    


if __name__ == '__main__':
    main()