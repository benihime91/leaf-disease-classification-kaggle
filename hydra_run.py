if __name__ == '__main__':
    """
    Use this script to run train the pytorch-lightning model
    To run this script we need to pass in two arguments to the 
    command line while running this script

    To run:
    python hydra_run.py \
        --config_dir {path to the main config directory} \
        --config_fname {the name of the config (usually the file name without the .yaml extension)}
    """
    import argparse
    from hydra.experimental import compose, initialize_config_dir
    from experiment import run
    import os

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir",
                        type=str,
                        help="directory where the config files the stored.",
                        required=True
                        )

    parser.add_argument("--config_fname",
                        type=str,
                        default="config",
                        help="name the parent config file without the .yaml extension.",
                        required=False
                        )

    # parse given arguments
    args = parser.parse_args()

    abs_config_dir = os.path.abspath(args.config_dir)
    # initialize hydra config
    with initialize_config_dir(config_dir=abs_config_dir, job_name="train_lightning_model"):
        # Run pytorch-lightning model
        config = compose(config_name=args.config_fname)
        # run model training
        run(config)
