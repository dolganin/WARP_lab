import yaml

class ConfigParser():
    # Initialization of the ConfigParser object
    def __init__(self, filename = None) -> None:
        # If a filename is provided, set it as base_name
        if filename is not None:
            self.base_name = filename 
        else:# If no filename is provided, set the default value
            self.base_name = "configs/base.yaml"
        # Call the method to parse the configuration file
        self.parse_config()

    # Method to parse the configuration file
    def parse_config(self):
        # Open the configuration file in read mode
        with open(self.base_name, 'r') as f:
            # Read the data from the file and convert it from YAML format to Python objects
            config = yaml.safe_load(f)
        # Save the configuration to an object attribute
        self.config = config
