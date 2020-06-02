from Utils.ConfigProvider import ConfigProvider
import os


if __name__ == "__main__":
    def main():
        print("hola")
        config = ConfigProvider.config()
        assert os.path.isdir(config.data.defective_examples_folder_path)
        assert os.path.isdir(config.data.non_defective_examples_folder_path)
    main()