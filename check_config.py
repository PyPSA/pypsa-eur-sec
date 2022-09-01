import yaml
from deepdiff import DeepDiff

if __name__ == '__main__':

    with open("config.default.yaml", 'r') as fp:
        config_old = yaml.safe_load(fp)

    with open("config.default.new.yaml", 'r') as fp:
        config_new = yaml.safe_load(fp)

    ddiff = DeepDiff(config_old, config_new, ignore_order=True)
    print(ddiff)
