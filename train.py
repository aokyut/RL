import argparse
from sac import sac
from torch.utils.tensorboard import SummaryWriter
from os.path import join
import json
from utils.memory import ReplayMemory
from base.trainer import gym_trainer
import gym

def get_default(name):
    with open(join(name, "default.json"), "r") as f:
        default_dict = json.load(f)
    return default_dict


models = {
    "sac": (sac.SAC, get_default("sac"))
}

def copy_param(args):
    config = {}
    for key, value in args.__dict__.items():
        if not callable(value):
            config[key] = value
    return config


def pull_model_and_writer(model_class, args):
    config = copy_param(args)
    if args.load:
        model = model_class(config)
        model.load()
        config = model.config
        for key, val in config.items():
            print(f"{key}: {val}")
        writer = SummaryWriter(log_dir=join(config["log_dir"], config["log_name"]),
                               purge_step=config["log_step"] // config["log_interval"])
        print(f"->resume from: {model.step}")
    else:
        for key, val in config.items():
            print(f"{key}: {val}")
        model = model_class(config)
        writer = SummaryWriter(log_dir=join(args.log_dir, args.log_name))
    memory = ReplayMemory(config["memory_size"])
    train_n = config["train_n"]
    return model, writer, memory, train_n
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()
    subparser.required = True


    for class_name, value in models.items():
        class_parser = subparser.add_parser(class_name)
        default_dict = value[1]
        for key, default_value in default_dict.items():
            if type(default_value) is bool:
                if default_value:
                    class_parser.add_argument(f"--{key}", action="store_false", default=True)
                else:
                    class_parser.add_argument(f"--{key}", action="store_true", default=False)
            else:
                class_parser.add_argument(f"--{key}", type=type(default_value), default=default_value)
        class_parser.set_defaults(func=lambda name: pull_model_and_writer(value[0], name))
            
    args = parser.parse_args()
    model, writer, memory, train_n = args.func(args)


    env = gym.make('Hopper-v3')
    gym_trainer(model=model, env=env, writer=writer, memory=memory, learn_n=train_n)