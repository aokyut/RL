import os
from os import path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

def categorical_sigmoid_policy(value, action_mask):
    return F.sigmoid(value) * action_mask

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def huber_error(td_error)-> torch.Tensor:
    x = torch.abs(td_error)
    return torch.square(x) / (x + 1)

def convert_network_grad_to_false(network):
    for param in network.parameters():
        param.requires_grad = False

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def save_model(step, model, name):
    model_dir = path.join("checkpoint", name)
    model_path = path.join(model_dir, f"{step}.pth")
    if not path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_path)

def _save_model(save_dir, save_name, model_name, model):
    model_dir = path.join(save_dir, save_name)
    model_path = path.join(model_dir, f"{model_name}.pth")
    print(f"save at {model_path}")
    if not path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_path)

# TODO: 入力を対話的に行えるようにする。
class ConfigParser:
    def __init__(self, name):
        self.dict = {}
        self.args = {}
        self.name = name
        self.setting_file = None
        self.setting_path = path.join(os.getcwd(), "setting.json")

    def add_parser(self, key, default, arg_type, description=""):
        self.dict[key] = (arg_type, default, description)
        assert(type(default) == arg_type)
    
    def parse(self):
        if self.setting_file is None:
            if path.exists(self.setting_path):
                with open(self.setting_path) as f:
                    setting_dict = json.load(f)
                self.setting_file = setting_dict
            else:
                self.setting_file = {}
        
        # setting.jsonの中に既に対象のconfigが存在するとき
        if self.name in self.setting_file:
            print("config file already exists. use this?")
            while True:
                try:
                    ans = input("y or n >>")
                    if ans is "y" or ans is "n":
                        break
                except Exception as e:
                    print(e)
            if ans is "y":
                self.open()
            else:
                self.init()
        else:
            self.init()
        self.save()
        return self.args

    def init(self):
        for key, item in self.dict.items():
            arg_type, default, description = item
            self.parse_value(arg_type, key, default, description, "init")
    
    def parse_value(self, arg_type, key, default, description, mode):
        if arg_type == bool:
            print(f"[{mode}] {key}={default}: {description}  input t or f")
            while True:
                try:
                    arg = str(input(">>"))
                    if arg == "t" or arg == "T":
                        self.args[key] = True
                        break
                    elif arg == "f" or arg == "F":
                        self.args[key] = False
                        break
                    elif arg == "":
                        self.args[key] = default
                        break
                except Exception as e:
                    print(e)
        else:
            print(f"[{mode}] {key}={default}: {description}")
            while True:
                try:
                    arg = input(">>")
                    if arg == "":
                        self.args[key] = default
                        break
                    else:
                        self.args[key] = arg_type(arg)
                        break
                except Exception as e:
                    print(e)

    def open(self):
        setting_dict = self.setting_file
        tar_setting = setting_dict[self.name]
        
        for key, item in self.dict.items():
            arg_type, default, description = item
            # 前回から変数が増えているとき
            if not key in tar_setting:
                self.parse_value(arg_type, key, default, description, "add")
                continue
            # 前回とは型が異なっているとき
            if type(tar_setting[key]) != arg_type:
                self.parse_value(arg_type, key, default, description, "modify")
                continue

            self.args[key] = tar_setting[key]

    def save(self):
        self.setting_file[self.name] = self.args
        with open(self.setting_path, 'wt') as f:
            json.dump(self.setting_file, f, indent=2)
        