import json
from typing import List, Dict, Union
import dill as pickle

UTF8 = "utf-8"


def load_json(path: str) -> Union[List[Dict], Dict]:
    with open(path, "r", encoding=UTF8) as file:
        data = json.load(file)
    return data


def write_json(data: Union[List, Dict[str, List]], path: str):
    with open(path, "w", encoding=UTF8) as file:
        json.dump(data, file, indent=2)


def read_lines_to_list(path: str) -> List[str]:
    with open(path, "r", encoding=UTF8) as file:
        return [line.strip() for line in file]


def write_list_data(path: str, data: List[str]):
    with open(path, "w", encoding=UTF8) as file:
        file.writelines([f"{elem}\n" for elem in data])


def write_pickle(path: str, data):
    with open(path, "wb") as file:
        pickle.dump(data, file)


def load_pickle(path: str):
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data
