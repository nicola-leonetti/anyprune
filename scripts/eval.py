import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg):
    print(f"Running on machine: {cfg.machine.name}")
    print(f"seed: {cfg.seed}")

if __name__ == "__main__":
    main()
