import argparse
import csv

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import MovementDataset
from models import MovementPredictor
from util.data import ROIS, get_data_array
from util.misc import grad_norm, save_model, set_seed


def get_data(t_width_s=0.05, regions=None):
    all_input, all_output = get_data_array(t_width_s=t_width_s)
    xtr, xts, ytr, yts = train_test_split(all_input, all_output, test_size=0.1, shuffle=True)
    tr_dataset = MovementDataset(list(zip(xtr, ytr)))
    ts_dataset = MovementDataset(list(zip(xts, yts)))

    return tr_dataset, ts_dataset


def train_one_epoch(model, optimizer, train_dataloader, log_writer, epoch, log_loss_every=5):

    losses = []
    for step, (x, y) in enumerate(train_dataloader):
        x, y = x.to(DEVICE).float(), y.to(DEVICE)
        y.masked_fill_(y < 0, 0)  # {-1, 1} -> {0, 1} for BCE loss

        optimizer.zero_grad()
        logits, _ = model(x)  # Model passes back attention weights, don't need here
        loss = F.binary_cross_entropy_with_logits(logits.view(-1), y.view(-1), reduction='mean')

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        gnorm = grad_norm(model)

        optimizer.step()

        if (step + 1) % log_loss_every == 0:
            losses.append(loss.item())
            log_writer.writerow(
                {'step': step, 'train_loss': losses[-1], 'grad_norm': gnorm.item()}
            )
            print(f"[Step {step}] Loss: {losses[-1]:.3f} Grad norm: {gnorm.item():.3f}")


@torch.no_grad()
def run_testing(model, test_dataloader):
    model.eval()
    preds, gt = [], []
    for step, (x, y) in enumerate(test_dataloader):
        x, y = x.to(DEVICE).float(), y.to(DEVICE)
        logits, _ = model(x)
        logits = logits.view(-1)

        pred = torch.zeros_like(logits).byte()
        pred.masked_fill_(logits < 0, -1)
        pred.masked_fill_(logits >= 0, 1)

        preds.extend(pred.cpu().numpy().tolist())
        gt.extend(y.cpu().numpy().tolist())

    return accuracy_score(gt, preds)


def main(args: argparse.Namespace):

    job_desc = (
        f'lr{args.learning_rate}_bs{args.batch_size}_wd{args.weight_decay}_'
        f'connx{int(args.use_connectome_weights)}_t{args.time_bin_width}'
    )
    print(f"Job identifier: {job_desc}")

    model = MovementPredictor(
        num_regions=len(ROIS),
        use_connectome_attn_weights=args.use_connectome_weights,
        hidden_dim=512,
        num_convs=1,
        attn_hidden_dim=256,
        t_in=int(6 / args.time_bin_width),  # 3 seconds total duration, divided by specified width
    ).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")

    optimizer = torch.optim.Adam(
        model.parameters(),
        args.learning_rate,
        weight_decay=args.weight_decay,
    )

    tr_dataset, ts_dataset = get_data(args.time_bin_width)
    train_dl = DataLoader(tr_dataset, batch_size=args.batch_size, num_workers=4)
    test_dl = DataLoader(ts_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    with (
        open(job_desc + '_' + args.train_log_path, 'w') as tr_log_fh,
        open(job_desc + '_' + args.test_log_path, 'w') as ts_log_fh,
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

    save_model(model, job_desc + '_' + args.model_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--use-connectome-weights', action='store_true')
    p.add_argument('-e', '--num-epochs', default=100, type=int)
    p.add_argument('-s', '--random-seed', default=42, type=int)
    p.add_argument('-b', '--batch-size', default=256, type=int)
    p.add_argument('-lr', '--learning-rate', default=1e-3, type=float, help='Learning rate')
    p.add_argument('-w', '--weight-decay', default=0, type=float, help='Weight decay')
    p.add_argument('-m', '--model-path', default='model.pt', type=str)
    p.add_argument('-t', '--time-bin-width', default=0.05, type=float, help='Time bin width (s)')
    p.add_argument('--train-log-path', default='train_logs.csv')
    p.add_argument('--test-log-path', default='test_logs.csv')

    return p.parse_args()


if __name__ == '__main__':
    ARGS = parse_args()
    set_seed(ARGS.random_seed)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    DEVICE = torch.device('cuda')
    main(ARGS)
