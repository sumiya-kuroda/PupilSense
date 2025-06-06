import defopt
from pupilsense.io import get_base_dir
from pupilsense.training.finetune_pupilsense import Finetune
from pathlib import Path

def main(*, model_path='', cfg_path=''):
    """Train model

    :param str model_path: file path to save a NEW pth file.
    :param str cfg_path: file path to load existing config yaml file.
    """
    print('---Finetune the PupilSense model---')

    finetune = Finetune()

    # Construct the full paths using the placeholders and the base directory
    if model_path=='':
        model_path = str(Path(get_base_dir()) / "models")
    train_json_path = str(Path(get_base_dir()) / 'dataset' / 'train_data.json')
    train_data_path = str(Path(get_base_dir()) / 'dataset' / 'train')
    finetune.set_config_train(model_path, train_json_path, train_data_path)

    # Save the configuration to a config.yaml file
    if cfg_path=='':
        cfg_path = str(Path(get_base_dir()) / "models" / "Config" / "config.yaml")
    finetune.save_cfg(cfg_path)

    # Start the training process
    print('Starting the training process')
    finetune.train()
    print('Training process finished!')

if __name__ == '__main__':
    defopt.run(main)