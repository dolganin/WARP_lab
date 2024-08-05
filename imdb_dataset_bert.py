import datasets
import itertools
from transformers import DistilBertTokenizer

class IMDBDataset():
    """A class for loading and preprocessing the IMDB dataset.

    Attributes:
        train (dict): A dictionary containing the training data (texts and labels).
        test (dict): A dictionary containing the test data (texts and labels).

    Args:
        config: Configuration yaml file
    """

    def __init__(self, config) -> None:
        """Initializes the class, loads the data, and performs preprocessing.

        Args:
            max_len_test: The maximum length of a sequence in the test data.
            max_len_train: The maximum length of a sequence in the training data.
        """

        # Load the training and test data, using only 4% of each split as a default. 
        self.train = datasets.load_dataset(config["dataset"]["dataset_name"], split="train", cache_dir="/workspace/datasets")
        self.test = datasets.load_dataset(config["dataset"]["dataset_name"], split="test", cache_dir="/workspace/datasets")
        self.token = DistilBertTokenizer.from_pretrained(config["reward_model_parameters"]["bert"])

        # Preprocess the data
        self.preprocess_data(config["dataset"]["test_max_len"], config["dataset"]["train_max_len"])

        # Create pairs for training and validating(e.g., positive and negative examples)
        self.data_train = self.create_pairs(self.train)
        self.data_test = self.create_pairs(self.test)       

    def encode_text(self, data):
        """Encodes text using the tokenizer."""
        return self.token(data, truncation=True, padding=True, return_tensors="pt")

    def preprocess_data(self, max_len_test, max_len_train) -> None:
        """Preprocesses the data: extracts text and labels, truncates sequences.

        Args:
            max_len_test: The maximum length of a sequence in the test data.
            max_len_train: The maximum length of a sequence in the training data.
        """

        # Extract text and labels
        self.train = {"data": self.train["text"], "label": self.train["label"]}
        self.test = {"data": self.test["text"], "label": self.test["label"]}

        # Truncate sequences to the specified length
        self.train["data"] = [self.encode_text(train_token[:max_len_train]) for train_token in self.train["data"]]
        self.test["data"] = [self.encode_text(test_token[:max_len_test]) for test_token in self.test["data"]]

    def create_pairs(self, data) -> None:
    
        """Creates pairs of positive and negative examples for training reward model."""

        # Find negative indices in the training data
        neg_samples_indices = [index for index, value in enumerate(data["label"]) if value == 0]
        # Find positive indices in the training data by difference between two sets
        pos_samples_indices = set(list(range(0, len(data["data"])))) - set(neg_samples_indices)

        # Calculate the zip of negative and positive examples 
        
        indices_perm = list(zip(neg_samples_indices, pos_samples_indices))

        data["data"] = [[data["data"][index[0]], data["data"][index[1]]] for index in indices_perm]
        data["label"] = [[data["label"][index[0]], data["label"][index[1]]] for index in indices_perm]

        performed_data = self.bert_proccessor(data=data)

        return performed_data
        #return data
    
    def bert_proccessor(self, data) -> None:
        """Preprocesses the data for the BERT model.

        This function prepares the data for input to the BERT model by creating
        dictionaries containing input_ids and attention_masks for chosen and rejected
        examples.
        """

        bert_tokens = []
        for index, pair in enumerate(data["data"]):
            label = data["label"][index]
            chosen_index = 0 if label[0] == 0 else 1
            rejected_index = 1 - chosen_index

            bert_tokens.append({
                "input_ids_chosen": pair[chosen_index]["input_ids"].squeeze(0),
                "attention_mask_chosen": pair[chosen_index]["attention_mask"].squeeze(0),
                "input_ids_rejected": pair[rejected_index]["input_ids"].squeeze(0),
                "attention_mask_rejected": pair[rejected_index]["attention_mask"].squeeze(0),
            })

        return bert_tokens

        
    def get_data(self):
        return self.data_train, self.data_test
    