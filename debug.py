import re
import os
import matplotlib.pyplot as plt

# read txt
def parse_log(log_path, num_processes=4, skip_first:int=10, ema=0.8):
    log_name = log_path.split("/")[-2]
    with open(log_path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        steps, losses = [], []
        for line in lines:
            # 使用正则表达式匹配step和loss
            match = re.search(r"step=(\d+).*Loss: ([\d.]+)", line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2)) * num_processes
                steps.append(step)
                losses.append(loss)
            else:
                pass
    if skip_first > 0:
        steps = steps[skip_first:]
        losses = losses[skip_first:]
    if ema > 0:
        ema_loss = []
        for i, loss in enumerate(losses):
            if i == 0:
                ema_loss.append(loss)
            else:
                ema_loss.append(ema * ema_loss[-1] + (1 - ema) * loss)
        losses = ema_loss
    return steps, losses


def draw_loss_curve(log_paths: list):
    model_losses = dict()
    for log_path in log_paths:
        log_name = log_path.split("/")[-2]
        steps, losses = parse_log(log_path)
        model_losses[log_name] = (steps, losses)
    # draw the loss curve
    plt.figure()
    for model, (steps, losses) in model_losses.items():
        plt.plot(steps, losses, label=model)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    # save the figure
    plt.savefig("loss_curve.png")
    
    
if __name__ == "__main__":
    models = ["002-DiT-S-2", "001-DiM-S-2", "003-DiM-S-2", "004-DiM-S-2"]
    log_paths = [f"results/{model}/log.txt" for model in models]
    draw_loss_curve(log_paths)