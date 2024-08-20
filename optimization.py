import torch

from tqdm import tqdm
from torchmetrics import Accuracy
from torchvision.transforms import v2


def train(epoch, accelerator, model, trainloader, criterion, optimizer, n_clases):
    model.train()

    running_loss = 0
    accuracy = Accuracy(task="multiclass", num_classes=n_clases).to(accelerator.device)
    
    mixup = v2.MixUp(alpha=1.0, num_classes=n_clases)
    cutmix = v2.CutMix(alpha=1.0, num_classes=n_clases)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    with tqdm(trainloader, unit='batch', desc='Train', disable=not accelerator.is_main_process) as tepoch:
        for idx, (inputs, targets) in enumerate(tepoch):
            optimizer.zero_grad()

            inputs, targets_mix = cutmix_or_mixup(inputs, targets)
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets_mix)
            accelerator.backward(loss)

            # Clip Gradient global norm & take step
            max_norm = 1.0  
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            # Metrics
            running_loss += loss.item()
            tepoch.set_postfix(loss=running_loss/(idx+1))
            _, predicted = outputs.max(1)
            accuracy.update(predicted, targets)

    epoch_acc = accuracy.compute().item()
    epoch_loss = running_loss/len(trainloader)
    if accelerator.is_main_process:
        print('Train Acc. ', "{:.4f}".format(epoch_acc))
    
    return epoch_loss


def validate(epoch, accelerator, model, testloader, criterion, n_clases):
    model.eval()

    running_loss = 0
    accuracy = Accuracy(task="multiclass", num_classes=n_clases).to(accelerator.device)

    with tqdm(testloader, unit='batch', desc='Val', disable=not accelerator.is_main_process) as tepoch:
        for inputs, targets in tepoch:
            with torch.no_grad():
                outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)

            all_predicted, all_targets = accelerator.gather_for_metrics((predicted, targets))
            accuracy.update(all_predicted, all_targets)

    epoch_acc = accuracy.compute().item()
    epoch_loss = running_loss/len(testloader)
    if accelerator.is_main_process:
        print('Val Acc. ', "{:.4f}".format(epoch_acc))
    return running_loss
    
