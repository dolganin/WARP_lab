from trl import RewardConfig, RewardTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from imdb_dataset_bert import IMDBDataset
from config_parse import ConfigParser


def main():
    """Main function for training a reward model."""

    # Load a pre-trained DistilBERT model for sequence classification
    reward_model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-cased", num_labels=1)

    # Load the corresponding tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-cased", use_fast=True)

    # Load configuration settings from a YAML file
    config = ConfigParser().config

    # Load and preprocess the IMDB dataset for training and validation
    train_dataset, validation_dataset = IMDBDataset(config=config).get_data()

    # Create a reward configuration object with parameters from the config file
    reward_config = RewardConfig(
        output_dir=config["general"]["output_dir"],  # Output directory for training artifacts
        remove_unused_columns=config["reward_model_parameters"]["remove_unused_columns"],  # Whether to remove unused columns
        per_device_train_batch_size=config["reward_model_parameters"]["per_device_train_batch_size"],  # Batch size for training
        num_train_epochs=config["reward_model_parameters"]["num_epochs"],  # Number of training epochs
        max_length=config["reward_model_parameters"]["max_length"],  # Maximum sequence length
        gradient_checkpointing=config["reward_model_parameters"]["gradient_checkpointing"],  # Whether to use gradient checkpointing
    )

    # Create a reward trainer object
    reward_trainer = RewardTrainer(
        model=reward_model,  # The reward model
        train_dataset=train_dataset,  # Training dataset
        eval_dataset=validation_dataset,  # Validation dataset
        tokenizer=tokenizer,  # Tokenizer for preprocessing text
        args=reward_config  # Training configuration
    )

    # Train the reward model
    reward_trainer.train()

    # Evaluate the trained model
    metrics = reward_trainer.evaluate()

    # Log evaluation metrics
    reward_trainer.log_metrics("eval", metrics)


if __name__ == "__main__":
    main()

