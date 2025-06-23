import json
import os

current_folder = os.path.dirname(os.path.abspath(__file__))


class Constants:
    def __init__(self, config_path: str = os.path.join(current_folder, "config.json")):
        """
        Initialize the Constants class.

        Args:
            config_path (str): Path to the configuration file.
        """
        self.config_path = config_path
        self.config_json = None

        self.load_config_from_file()

    def load_config_from_file(self) -> dict:
        """
        Load the configuration file.

        Returns:
            dict: The loaded configuration as a dictionary.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as file:
                self.config_json = json.load(file)
        else:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        return self.config_json

    def load_config_from_json(self, json_data: dict) -> dict:
        """
        Load the configuration from a dictionary.

        Args:
            json_data (dict): The configuration data as a dictionary.

        Returns:
            dict: The loaded configuration as a dictionary.
        """

        for key, value in json_data.items():
            if isinstance(value, dict):
                if key not in self.config_json:
                    self.config_json[key] = {}
                self.config_json[key].update(value)
            else:
                self.config_json[key] = value

        self.save_config()
        return self.config_json

    def get_config(self) -> dict:
        """
        Get the loaded configuration.

        Returns:
            dict: The loaded configuration as a dictionary.
        """
        return self.config_json

    def get_value(self, key: str, default=None) -> any:
        """
        Get a value from the loaded configuration.

        Args:
            key (str): The key to retrieve the value for. Supports nested keys separated by dots.
            default (any): The default value to return if the key is not found.

        Returns:
            any: The value associated with the key, or the default value if the key is not found.
        """
        if "." in key:
            keys = key.split(".")
            value = self.config_json
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        return self.config_json.get(key, default)

    def set_value(self, key: str, value) -> None:
        """
        Set a value in the loaded configuration.

        Args:
            key (str): The key to set the value for.
            value (any): The value to set.
        """
        self.config_json[key] = value
        with open(self.config_path, "w") as file:
            json.dump(self.config_json, file, indent=4)

    def save_config(self) -> None:
        """
        Save the current configuration to the file.
        """
        with open(self.config_path, "w") as file:
            json.dump(self.config_json, file, indent=4)

    def __getitem__(self, item):
        """
        Allow access to class attributes using square brackets.

        Args:
            item (str): The key to retrieve the value for.

        Returns:
            any: The value associated with the key.
        """
        value = self.get_value(item)
        if value is None:
            raise KeyError(f"Key '{item}' not found in configuration.")
        return value


constants = Constants()
constants.load_config_from_file()

if __name__ == "__main__":
    # Example usage of the loaded configuration
    print(constants["NetworkTableConstants.server_address"])
