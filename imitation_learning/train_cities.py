import os
import time
import torch
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from dataset import (
    CityLabel,
    UnitLabel,
    to_unit_id,
    show_action_count,
    create_dataset_from_json,
)


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


# ======================
# Dataset
# ======================


NORTH_LABEL = UnitLabel.NORTH.value
EAST_LABEL = UnitLabel.EAST.value
SOUTH_LABEL = UnitLabel.SOUTH.value
WEST_LABEL = UnitLabel.WEST.value
TRANSFER_NORTH_LABEL = UnitLabel.TRANSFER_NORTH.value
TRANSFER_EAST_LABEL = UnitLabel.TRANSFER_EAST.value
TRANSFER_SOUTH_LABEL = UnitLabel.TRANSFER_SOUTH.value
TRANSFER_WEST_LABEL = UnitLabel.TRANSFER_WEST.value


def get_aug(p_rot=0.5, p_flip=0.5):
    # rotate
    rot = None
    rot_map = {}
    if np.random.random() < p_rot:
        rot = np.random.randint(low=1, high=4)
        if rot == 1:
            rot_map = {
                NORTH_LABEL: EAST_LABEL,
                EAST_LABEL: SOUTH_LABEL,
                SOUTH_LABEL: WEST_LABEL,
                WEST_LABEL: NORTH_LABEL,
                TRANSFER_NORTH_LABEL: TRANSFER_EAST_LABEL,
                TRANSFER_EAST_LABEL: TRANSFER_SOUTH_LABEL,
                TRANSFER_SOUTH_LABEL: TRANSFER_WEST_LABEL,
                TRANSFER_WEST_LABEL: TRANSFER_NORTH_LABEL,
            }
        elif rot == 2:
            rot_map = {
                NORTH_LABEL: SOUTH_LABEL,
                SOUTH_LABEL: NORTH_LABEL,
                EAST_LABEL: WEST_LABEL,
                WEST_LABEL: EAST_LABEL,
                TRANSFER_NORTH_LABEL: TRANSFER_SOUTH_LABEL,
                TRANSFER_SOUTH_LABEL: TRANSFER_NORTH_LABEL,
                TRANSFER_EAST_LABEL: TRANSFER_WEST_LABEL,
                TRANSFER_WEST_LABEL: TRANSFER_EAST_LABEL,
            }
        elif rot == 3:
            rot_map = {
                NORTH_LABEL: WEST_LABEL,
                WEST_LABEL: SOUTH_LABEL,
                SOUTH_LABEL: EAST_LABEL,
                EAST_LABEL: NORTH_LABEL,
                TRANSFER_NORTH_LABEL: TRANSFER_WEST_LABEL,
                TRANSFER_WEST_LABEL: TRANSFER_SOUTH_LABEL,
                TRANSFER_SOUTH_LABEL: TRANSFER_EAST_LABEL,
                TRANSFER_EAST_LABEL: TRANSFER_NORTH_LABEL,
            }
        else:
            raise ValueError(rot)

    # flip
    flip = None
    flip_map = {}
    if np.random.random() < p_flip:
        flip = np.random.randint(low=1, high=3)
        if flip == 2:
            flip_map = {
                NORTH_LABEL: SOUTH_LABEL,
                SOUTH_LABEL: NORTH_LABEL,
                TRANSFER_NORTH_LABEL: TRANSFER_SOUTH_LABEL,
                TRANSFER_SOUTH_LABEL: TRANSFER_NORTH_LABEL,
            }
        elif flip == 1:
            flip_map = {
                EAST_LABEL: WEST_LABEL,
                WEST_LABEL: EAST_LABEL,
                TRANSFER_EAST_LABEL: TRANSFER_WEST_LABEL,
                TRANSFER_WEST_LABEL: TRANSFER_EAST_LABEL,
            }
        else:
            raise ValueError(flip)

    action_map = {x.value: x.value for x in UnitLabel}
    if rot_map:
        action_map = {k: rot_map.get(v, v) for k, v in action_map.items()}
    if flip_map:
        action_map = {k: flip_map.get(v, v) for k, v in action_map.items()}

    return rot, flip, action_map


def aug(matrix, rot, flip):
    if rot is None and flip is None:
        return matrix
    if rot is not None:
        matrix = np.rot90(matrix, k=rot, axes=(1, 2))
    if flip is not None:
        matrix = np.flip(matrix, axis=flip)
    return matrix.copy()


RESOURCE_TO_LAYER = {"wood": 16, "coal": 17, "uranium": 18}


def make_input(obs, add_aug=False):
    player = obs["index"]
    step = obs["step"]
    updates = obs["updates"]
    width = obs["width"]
    height = obs["height"]

    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    team_to_num_city = defaultdict(int)

    b = np.zeros((20, 32, 32), dtype=np.float32)
    gf = np.zeros((7, 4, 4), dtype=np.float32)

    for update in updates:
        strs = update.split(" ")
        input_identifier = strs[0]

        if input_identifier == "u":
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])

            # Units
            team = int(strs[2])
            unit_id = to_unit_id(strs[3])
            cooldown = float(strs[6])
            idx = (team - player) % 2 * 6
            b[idx : idx + 6, x, y] = (
                1,
                cooldown / 6,
                (wood + coal + uranium) / 100,
                wood / 100,
                coal / 100,
                uranium / 100,
            )

        elif input_identifier == "ct":
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            idx = 12 + (team - player) % 2 * 2
            b[idx : idx + 2, x, y] = (1, cities[city_id])
            team_to_num_city[team] += 1

        elif input_identifier == "r":
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[RESOURCE_TO_LAYER[r_type], x, y] = amt / 800

        elif input_identifier == "rp":
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            gf[(team - player) % 2, :] = min(rp, 200) / 200

        elif input_identifier == "c":
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            cities[city_id] = min(fuel / lightupkeep, 10) / 10

    # Map Size
    b[19, x_shift : 32 - x_shift, y_shift : 32 - y_shift] = 1
    # b = aug(b, rot, flip)

    # Day/Night Cycle
    is_day = (step % 40) < 30
    gf[2, :] = is_day
    gf[3, :] = (step % 40) / 30 if is_day else ((step % 40) - 30) / 10
    # Turns
    gf[4, :] = step / 360
    # Num city
    gf[5, :] = team_to_num_city[player] / (width * height)
    gf[6, :] = team_to_num_city[1 - player] / (width * height)

    return b, gf, x_shift, y_shift


class LuxDataset(Dataset):
    def __init__(self, obses, samples, add_aug=False):
        self.obses = obses
        self.samples = samples
        self.add_aug = add_aug

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i, x0, y0, label = self.samples[idx]
        state, gf, x_shift, y_shift = make_input(self.obses[i], self.add_aug)

        return state, gf, x0 + x_shift, y0 + y_shift, label


def extract_samples(obses, field):
    samples = []
    for i, obs in enumerate(tqdm(obses)):
        actions = obs[field]
        for x, y, lb in actions:
            needed = True
            if lb == CityLabel.NOTHING.value and random.random() > 0.025:
                needed = False
            if needed:
                samples.append((i, x, y, lb))
    return samples


# ======================
# Model
# based on
# https://github.com/milesial/Pytorch-UNet/tree/master/unet
# https://www.kaggle.com/c/lux-ai-2021/discussion/289540
# ======================


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AddGlobal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, gf):
        return torch.cat([x, gf], dim=1)


class UNet(nn.Module):
    def __init__(self, n_channels=20, n_classes=3, n_global=7, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        self.add_global = AddGlobal()
        self.up1 = Up(512 + n_global, 256, bilinear)
        self.up2 = Up(256 + 128, 128, bilinear)
        self.up3 = Up(128 + 64, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, gf):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.add_global(x4, gf)
        x = self.up1(x5, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x


# ======================
# Train
# ======================


criterion = nn.CrossEntropyLoss()


def get_predictions(y, x0, y0):
    return y[range(len(y)), :, x0, y0]


def get_acc(policy, label, label_to_acc):
    correct = 0
    total = 0
    for p, lb in zip(policy, label):
        with_action = lb.any(axis=0)
        for x, y in zip(*torch.where(with_action)):
            _p = p[:, x, y]
            _lb = lb[:, x, y]
            _p = torch.argmax(_p)
            _lb = torch.argmax(_lb)

            correct += _p == _lb
            total += 1

            label_to_acc[int(_lb)][0] += int(_p == _lb)
            label_to_acc[int(_lb)][1] += 1

    return correct, total


def train_model(
    model, dataloaders_dict, optimizer, scheduler, num_epochs, model_name="model"
):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.cuda()

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_acc = 0
            coorect = 0
            total = 0

            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):
                states = item[0].cuda().float()
                gf = item[1].cuda().float()
                x0 = item[2].cuda().long()
                y0 = item[3].cuda().long()
                actions = item[4].cuda().long()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    policy = model(states, gf)
                    preds = get_predictions(policy, x0, y0)
                    loss = criterion(preds, actions)
                    _, preds = torch.max(preds, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * len(policy)
                    epoch_acc += torch.sum(preds == actions.data)

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            epoch_acc = epoch_acc.double() / data_size

            time.sleep(10)
            print(
                f"Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}"
            )

        if epoch_acc > best_acc:
            traced = torch.jit.trace(
                model.cpu(),
                example_inputs=(torch.rand(1, 20, 32, 32), torch.rand(1, 7, 4, 4)),
            )
            model_path = f"{model_name}.pth"
            print(f"Saving model to `{model_path}`.")
            traced.save(model_path)
            best_acc = epoch_acc


def main(episode_dir="episodes", model_name="model"):
    seed_everything(42)

    obses = create_dataset_from_json(
        episode_dir, num_episodes=None, team_name="Toad Brigade"
    )
    show_action_count(obses)

    model = UNet()  # torch.jit.load(f'{model_name}.pth')

    ep_ids = list(set(x["episode_id"] for x in obses))
    train_ep, val_ep = train_test_split(ep_ids, test_size=0.1, random_state=42)
    samples = extract_samples(obses, "city_actions")
    train = [x for x in samples if obses[x[0]]["episode_id"] in train_ep]
    val = [x for x in samples if obses[x[0]]["episode_id"] in val_ep]

    batch_size = 64
    train_loader = DataLoader(
        LuxDataset(obses, train), batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        LuxDataset(obses, val), batch_size=batch_size, shuffle=False, num_workers=0
    )
    dataloaders_dict = {"train": train_loader, "val": val_loader}

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_model(model, dataloaders_dict, optimizer, None, num_epochs=2, model_name=model_name)
    for lr in (1e-4, 1e-5, 1e-6):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        train_model(model, dataloaders_dict, optimizer, None, num_epochs=1, model_name=model_name)


if __name__ == "__main__":
    main(episode_dir="episodes", model_name="model")
