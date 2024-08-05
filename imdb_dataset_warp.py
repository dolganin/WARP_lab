import datasets

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

        # Preprocess the data
        self.preprocess_data(config["dataset"]["test_max_len"], config["dataset"]["train_max_len"])

    def preprocess_data(self, max_len_test, max_len_train) -> None:
        """Preprocesses the data: extracts text and labels, truncates sequences.

        Args:
            max_len_test: The maximum length of a sequence in the test data.
            max_len_train: The maximum length of a sequence in the training data.
        """

        # Truncate sequences to the specified length
        train = [train_token[:30] for train_token in self.train["text"]]
        test = [test_token[:20] for test_token in self.test["text"]]
        self.train = train
        self.test = test

    def get_data(self):
        return self.train, self.test
    