if __name__ == '__main__':
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

    # Run pytorch-lightning model
    abs_config_dir = os.path.abspath(args.config_dir)
    with initialize_config_dir(config_dir=abs_config_dir):
        config = compose(config_name=args.config_fname)
        # run model training
        run(config)
