import logging

import torch
from tqdm import trange

from src.modules import WhisperForDiarization, get_loader
from src.utils import hp

logger = logging.getLogger()
fh = logging.FileHandler(hp.log_path, mode="a", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(name)s-%(levelname)s %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

if hp.device == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(hp.device)

train_dataset = get_loader(
    split="train",
    batch_size=hp.train_batch_size,
    shuffle=True,
)
test_dataset = get_loader(
    split="test",
    batch_size=hp.test_batch_size,
    shuffle=False,
)

model = WhisperForDiarization()

optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)

for epoch in trange(hp.num_epoches, colour="green"):
    idx = 0
    for input_features, labels in train_dataset:
        idx += 1
        outputs = model(input_features=input_features, labels=labels)

        logits: torch.Tensor = outputs.logits
        loss: torch.Tensor = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.info(
            f"epoch : {epoch}, batch: {idx}, loss_avg : {loss.item() / len(input_features):.2f}"
        )

torch.save(model.state_dict(), hp.save_path)
