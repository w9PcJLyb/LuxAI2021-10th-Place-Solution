import json
import numpy as np
from enum import IntEnum
from pathlib import Path
from tqdm.notebook import tqdm
from collections import defaultdict


class UnitLabel(IntEnum):
    CENTER = 5
    NORTH = 0
    SOUTH = 1
    WEST = 2
    EAST = 3
    BUILD = 4
    TRANSFER_NORTH = 6
    TRANSFER_SOUTH = 7
    TRANSFER_WEST = 8
    TRANSFER_EAST = 9

    def __repr__(self):
        return self.name

    @classmethod
    def from_direction(cls, d):
        if d == "c":
            return UnitLabel.CENTER
        elif d == "n":
            return UnitLabel.NORTH
        elif d == "s":
            return UnitLabel.SOUTH
        elif d == "w":
            return UnitLabel.WEST
        elif d == "e":
            return UnitLabel.EAST
        else:
            raise ValueError(f"Unknown direction {d}")

    @classmethod
    def from_vector(cls, v):
        if v == (0, 0):
            return UnitLabel.CENTER
        elif v == (0, -1):
            return UnitLabel.NORTH
        elif v == (0, 1):
            return UnitLabel.SOUTH
        elif v == (-1, 0):
            return UnitLabel.WEST
        elif v == (1, 0):
            return UnitLabel.EAST
        else:
            raise ValueError(f"Unknown vector {v}")

    def transfer(self):
        if self == UnitLabel.NORTH:
            return UnitLabel.TRANSFER_NORTH
        elif self == UnitLabel.SOUTH:
            return UnitLabel.TRANSFER_SOUTH
        elif self == UnitLabel.WEST:
            return UnitLabel.TRANSFER_WEST
        elif self == UnitLabel.EAST:
            return UnitLabel.TRANSFER_EAST
        else:
            raise ValueError(f"Unsupported transfer to {self}")


class CityLabel(IntEnum):
    BUILD_WORKER = 0
    RESEARCH = 1
    NOTHING = 2


def to_unit_id(unit_str):
    return int(unit_str.split("_")[1])


def to_label(action_str, unit_id_to_info, episode_id, step):
    strs = action_str.split(' ')
    command = strs[0]
    unit_id = to_unit_id(strs[1])
    if command == 'm':
        direction = strs[2]
        label = UnitLabel.from_direction(direction).value
    elif command == 'bcity':
        label = UnitLabel.BUILD.value
    elif command == "t":
        unit = unit_id_to_info[unit_id]
        target_id = to_unit_id(strs[2])
        target = unit_id_to_info[target_id]
        x0, y0 = unit["x"], unit["y"]
        x, y = target["x"], target["y"]
        dc = (x - x0, y - y0)
        if not unit["cargo"]:
            label = None
        else:
            try:
                label = UnitLabel.from_vector(dc).transfer().value
            except ValueError:
                label = None
    else:
        label = None

    return unit_id, label


def depleted_resources(obs):
    for u in obs['updates']:
        if u.split(' ')[0] == 'r':
            return False
    return True


def create_dataset_from_json(episode_dir, team_name=None, num_episodes=None):
    obses = []

    episodes = [path for path in Path(episode_dir).glob('*.json') if 'output' not in path.name]
    if num_episodes is not None:
        episodes = episodes[:num_episodes]

    for filepath in tqdm(episodes):
        with open(filepath) as f:
            json_load = json.load(f)

        ep_id = json_load['info']['EpisodeId']
        index = np.argmax([r or 0 for r in json_load['rewards']])
        agent_name = json_load['info']['TeamNames'][index]
        if team_name is not None and agent_name != team_name:
            continue

        obses += pars_game_json(json_load)

    return obses


def pars_game_json(data):
    ep_id = data['info']['EpisodeId']
    index = np.argmax([r or 0 for r in data['rewards']])
    steps = data['steps']
    obses = []
    for i, (step, next_step) in enumerate(zip(steps[:-1], steps[1:])):
        if step[index]['status'] == 'ACTIVE':
            actions = next_step[index]['action']
            obs = step[0]['observation']

            # Read Updates
            unit_id_to_info = {}
            city_id_to_info = {}
            for row in obs["updates"]:
                if row == "D_DONE":
                    break

                row = row.split(" ")
                input_identifier = row[0]

                if input_identifier == "u":
                    unit_type = int(row[1])
                    team = int(row[2])
                    unit_id = to_unit_id(row[3])
                    x = int(row[4])
                    y = int(row[5])
                    cooldown = float(row[6])
                    wood = int(row[7])
                    coal = int(row[8])
                    uranium = int(row[9])

                    unit_id_to_info[unit_id] = {
                        "x": x,
                        "y": y,
                        "team": team,
                        "cooldown": cooldown,
                        "cargo": wood + coal + uranium,
                    }

                elif input_identifier == "c":
                    team = int(row[1])
                    city_id = row[2]
                    fuel = float(row[3])
                    light_up_keep = float(row[4])

                    city_id_to_info[city_id] = {
                        "team": team,
                        "fuel": fuel,
                        "light_up_keep": light_up_keep,
                        "tiles": []
                    }

                elif input_identifier == "ct":
                    city_id = row[2]
                    x = int(row[3])
                    y = int(row[4])
                    cooldown = float(row[5])

                    city_id_to_info[city_id]["tiles"].append(
                        {"x": x, "y": y, "cooldown": cooldown}
                    )

            tiles_can_act = set()
            for city_info in city_id_to_info.values():
                if city_info["team"] == index:
                    for tile in city_info["tiles"]:
                        if tile["cooldown"] < 1:
                            tiles_can_act.add((tile["x"], tile["y"]))

            # Read actions
            unit_actions = []
            city_actions = []
            units_with_actions = set()
            tiles_with_actions = set()
            for action_command in actions:
                command = action_command.split(" ")[0]

                # Units
                if command in ("bcity", "m", "t"):
                    unit_id, label = to_label(action_command, unit_id_to_info, ep_id, i)

                    unit_info = unit_id_to_info[unit_id]

                    team = unit_info["team"]
                    if team != index:
                        continue

                    units_with_actions.add(unit_id)

                    can_act = unit_info["cooldown"] < 1
                    if not can_act and label is not None:
                        print(f"Warning! Unit move but can't! ep_id={ep_id}, step={i}")

                    can_build = unit_info["cargo"] >= 100
                    if not can_build and label == UnitLabel.BUILD.value:
                        print(f"Warning! Unit build but can't! ep_id={ep_id}, step={i}")

                    if label is not None:
                        unit_actions.append((unit_id, label))

                # City
                elif command in ("bw", "bc", "r"):
                    _, x, y = action_command.split(" ")
                    x, y = int(x), int(y)
                    if (x, y) in tiles_can_act:
                        if command == "r":
                            label = CityLabel.RESEARCH.value
                        else:
                            label = CityLabel.BUILD_WORKER.value
                        city_actions.append((x, y, label))
                        tiles_with_actions.add((x, y))

            # Units without action
            for unit_id, unit_info in unit_id_to_info.items():
                if unit_info["team"] != index:
                    continue

                if unit_info["cooldown"] < 1 and unit_id not in units_with_actions:
                    unit_actions.append((unit_id, UnitLabel.CENTER.value))

            # City Tiles without action
            for tile in tiles_can_act:
                if tile not in tiles_with_actions:
                    x, y = tile
                    city_actions.append((x, y, CityLabel.NOTHING.value))

            if city_actions:
                obses.append(
                    {
                        "index": index,
                        "width": obs["width"],
                        "height": obs["height"],
                        "step": obs["step"],
                        "episode_id": ep_id,
                        "updates": tuple(obs["updates"]),
                        "unit_actions": unit_actions,
                        "city_actions": city_actions,
                    }
                )

    return obses


def show_action_count(obses):
    print(f'Number of observations: {len(obses)}')

    unit_action_to_count = defaultdict(int)
    city_action_to_count = defaultdict(int)
    for obs in obses:
        for _, label in obs["unit_actions"]:
            unit_action_to_count[label] += 1
        for _, _, label in obs["city_actions"]:
            city_action_to_count[label] += 1

    print("Unit lables: ")
    for label in UnitLabel:
        print(
            f" - {label.value} {label.name} - "
            f"{unit_action_to_count.get(label.value, 0)} samples"
        )

    print("City lables: ")
    for label in CityLabel:
        print(
            f" - {label.value} {label.name} - "
            f"{city_action_to_count.get(label.value, 0)} samples"
        )
