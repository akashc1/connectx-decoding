import argparse
import csv

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models import MovementPredictor
from util.misc import set_seed, save_model
from util.data import get_data_array, ROIS
from dataset import MovementDataset


def get_data(regions=None, batch_size=16):
    all_input, all_output = get_data_array()
    xtr, xts, ytr, yts = train_test_split(all_input, all_output, test_size=0.1, shuffle=True)
    tr_dataset = MovementDataset(list(zip(xtr, ytr)))
    ts_dataset = MovementDataset(list(zip(xts, yts)))

    return tr_dataset, ts_dataset


def train_one_epoch(model, optimizer, train_dataloader, log_writer, epoch, log_loss_every=5):

    losses = []
    # _x, _y = next(iter(train_dataloader))
    for step, (x, y) in enumerate(tqdm(train_dataloader)):
    # for step in range(3000):
        # if _x is None and _y is None:
        #     _x, _y = x, y

        # x, y = _x.clone().to(DEVICE), _y.clone().to(DEVICE)
        # print(x[0, 0, 0])
        x, y = x.to(DEVICE).float(), y.to(DEVICE)
        y.masked_fill_(y < 0, 0)

        optimizer.zero_grad()
        logits = model(x).view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y.view(-1), reduction='mean')

        loss.backward()
        optimizer.step()

        if (step + 1) % log_loss_every == 0:
            losses.append(loss.item())
            grad_norm = torch.norm(
                torch.cat([p.view(-1) for p in model.parameters()])
            )
            log_writer.writerow(
                {'step': step, 'train_loss': losses[-1], 'grad_norm': grad_norm.item()}
            )
            print(f"[Step {step}] Loss: {losses[-1]:.3f} Grad norm: {grad_norm.item():.3f}")


@torch.no_grad()
def run_testing(model, test_dataloader):
    model.eval()
    preds, gt = [], []
    for step, (x, y) in enumerate(tqdm(test_dataloader, desc='Test loop')):
        x, y = x.to(DEVICE).float(), y.to(DEVICE)
        logits = model(x).view(-1)

        pred = torch.zeros_like(logits).byte()
        pred.masked_fill_(logits < 0, -1)
        pred.masked_fill_(logits >= 0, 1)

        preds.extend(pred.cpu().numpy().tolist())
        gt.extend(y.cpu().numpy().tolist())

    return accuracy_score(gt, preds)


def main(args: argparse.Namespace):
    set_seed(args.random_seed)
    model = MovementPredictor(
        num_regions=len(ROIS),
        use_connectome_attn_weights=args.use_connectome_weights,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        args.learning_rate,
        weight_decay=args.weight_decay,
    )

    tr_dataset, ts_dataset = get_data()
    train_dl = DataLoader(tr_dataset, batch_size=args.batch_size, num_workers=4)
    test_dl = DataLoader(ts_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    with (
        open(args.train_log_path, 'w') as tr_log_fh,
        open(args.test_log_path, 'w') as ts_log_fh,
    ):
        tr_writer = csv.DictWriter(tr_log_fh, ['step', 'train_loss', 'grad_norm'])
        ts_writer = csv.DictWriter(ts_log_fh, ['epoch', 'test_accuracy'])
        tr_writer.writeheader()
        ts_writer.writeheader()

        for e in tqdm(range(args.num_epochs)):
            model.train()
            train_one_epoch(model, optimizer, train_dl, tr_writer, e)

            acc = run_testing(model, test_dl)
            ts_writer.writerow({'epoch': e, 'test_accuracy': acc})
            print(f"Epoch {e}: test accuracy {acc:.3f}")

    save_model(model, args.model_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--use-connectome-weights', action='store_true')
    p.add_argument('-e', '--num-epochs', default=20, type=int)
    p.add_argument('-s', '--random-seed', default=42, type=int)
    p.add_argument('-b', '--batch-size', default=64, type=int)
    p.add_argument('-lr', '--learning-rate', default=3e-4, type=float, help='Learning rate')
    p.add_argument('-w', '--weight-decay', default=0, type=float, help='Weight decay')
    p.add_argument('-m', '--model-path', default='model.pt', type=str)
    p.add_argument('--train-log-path', default='train_logs.csv')
    p.add_argument('--test-log-path', default='test_logs.csv')

    return p.parse_args()


if __name__ == '__main__':
    ARGS = parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    DEVICE = torch.device('cuda')
    main(ARGS)
