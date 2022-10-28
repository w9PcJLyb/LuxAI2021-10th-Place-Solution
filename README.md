# LuxAI2021 10th Place Solution

Lux AI Competition - https://www.kaggle.com/c/lux-ai-2021

The architecture I used was inspired by [this post](https://www.kaggle.com/c/lux-ai-2021/discussion/289540) 

Major changes:

- Separate model to predict city actions {BUILD_WORKER, RESEARCH, NOTHING}
- Transfer actions for units {TRANSFER_NORTH, TRANSFER_SOUTH, TRANSFER_WEST, TRANSFER_EAST}
- Augmentations (Flip, Rotate)

`agents/IL_1689.tar.gz` - My best agent, 10th place in the competition

How to train the agent: 

1. download episodes, example: https://www.kaggle.com/code/robga/simulations-episode-scraper-match-downloader
2. train the unit model: `python3 imitation_learning/train_units.py --episode_dir "path_to_episodes"`
3. train the city model: `python3 imitation_learning/train_cities.py --episode_dir "path_to_episodes"`
4. copy `agents/IL_1689.tar.gz` and update the models there.
