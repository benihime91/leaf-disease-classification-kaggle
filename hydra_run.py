if __name__ == '__main__':
    import argparse
    from hydra.experimental import compose, initialize
    from omegaconf import DictConfig, OmegaConf
    from experiment import run

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
    with initialize(config_path=args.config_dir, job_name="run_model"):
        config = compose(config_name=args.config_fname)
        run(config)
