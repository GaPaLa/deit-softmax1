# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import numpy as np
import matplotlib.pyplot as plt
import pickle




norm_bins = 1000
bins_ = np.insert(np.logspace(np.log10(1),np.log10(norm_bins), norm_bins),   0,0)


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None, softmax1=False):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with torch.cuda.amp.autocast():
            outputs = model(x=samples, softmax1=softmax1)
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # --- plot average patch norms for each layer this batch
        if i%1000==0:
            del outputs
            with torch.no_grad():
                _, attn_maps, norms = model(x=samples, output_patchnorms=True, output_attentions=True, softmax1=softmax1)

                with open(f'/mnt/imgnet/job/layernorms_histograms_{epoch}_{i}.npy','wb') as file:
                    # keep track of norms for batch
                    layernorms_histograms = np.zeros([norm_bins, len(model.blocks)+1]) # +1 for embedding layer
                    for l, normlayer in enumerate(norms): # normlayer has shape [batch_size, tokens]
                        hist, _ = np.histogram(normlayer.reshape([-1]).half().cpu().numpy(), bins=bins_, range=(0, norm_bins)) # get histogram of norms for each layer
                        layernorms_histograms[:,l] = hist
                    np.save(file, layernorms_histograms)
                with open(f'/mnt/imgnet/job/attn_map_{epoch}_{i}.npy','wb') as file:
                    np.save(file, attn_maps[-1][0].mean(dim=0).half().cpu().numpy())
                
                del layernorms_histograms, normlayer, file, attn_maps, norms
                torch.cuda.empty_cache()


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def evaluate(data_loader, model, device, plot_norms=False, plot_att=False, softmax1=False, epoch=0):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    
    layernorms_histograms = np.zeros([norm_bins, len(model.blocks)+1]) #+1 for embedding layer
    for i, (images, target) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output, attn_maps, norms = model(images, output_attentions=plot_att, output_patchnorms=plot_norms, softmax1=softmax1)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


        # --- plot norms, attn
        norms = [normlayer.cpu().numpy() for normlayer in norms] # remember! norms includes the outputs of all alyers, but also the embedded layer norm!!!
        attn_maps = [attmap.half().mean(dim=1).cpu().numpy() for attmap in attn_maps] # average attention across heads - to much data otherwise
        # eval uses SequentialDataLoader so we should be safe just saving the first N images. They may not be in the same order fif distributed but should still be the same ones. just check if its the same across multiple traiing runs for epoch 0
        if i==0: # only save the first batch of images for eval - these will all be of the same class i think but still.

            # save attentions for first batch of eval
            with open('/mnt/imgnet/job/'+str(epoch)+'_attn.npy', 'wb') as file:
                pickle.dump(attn_maps, file)
            # save norms for first batch of eval
            with open('/mnt/imgnet/job/'+str(epoch)+'_norms.npy', 'wb') as file:
                pickle.dump(norms, file)
            
            # plot images for first batch of eval
            #plt.imshow(images[0].permute(1,2,0).cpu().numpy())
            #plt.savefig(f'/mnt/imgnet/job/{i, epoch}_image0.png')
            
            # plot last layer attn map for first element of eval
            plt.imshow(attn_maps[-1][0]) # last layer, first batch, average of heads
            plt.savefig(f'/mnt/imgnet/job/{epoch}_attn.png')

            # keep track of norms for whole eval
            for l, normlayer in enumerate(norms): # normlayer has shape [batch_size, tokens]
                hist, edges = np.histogram(   normlayer.reshape([-1]),    bins=bins_,     range=(0, norm_bins)) # get histogram of norms for each layer
                layernorms_histograms[:,l] = hist/(batch_size * norms[0].shape[1])
        else:
            # keep track of norms for whole eval
            for l, normlayer in enumerate(norms): # normlayer has shape [batch_size, tokens]
                hist, edges = np.histogram(normlayer.reshape([-1]), bins=bins_, range=(0, norm_bins)) # get histogram of norms for each layer
                layernorms_histograms[:,l] += hist/(batch_size * norms[0].shape[1])

    # get average norm frequency # replicating plotting style from "ViTs Need Registers" paper
    layernorms_histograms /= i
    # plot average norms heatmap
    y_values = np.logspace(np.log10(1), np.log10(norm_bins), norm_bins)
    y_tick_indices = [3, 30, 300]
    y_ticks = [np.argmin((y_values-target)**2) for target in y_tick_indices]
    print(y_ticks)
    print(layernorms_histograms.shape)
    #plt.imshow(np.log10(layernorms_histograms+0.01), cmap='hot', origin='lower', aspect='auto', extent=[0, len(norms), 0, norm_bins], interpolation='nearest', vmin=(10e-3)-0.1, vmax=(10e-1)+0.1)
    plt.imshow(np.log10(layernorms_histograms+0.01), cmap='hot', origin='lower', aspect='auto', extent=[0, len(norms), 0, norm_bins], interpolation='nearest', vmin=np.log10(0+0.01), vmax=np.log10(0.8+0.01) )
    plt.yticks(y_ticks, ['3', '30', '300'])

    colorbar_ticks = np.array([10e-3, 10e-2, 10e-1])
    colorbar = plt.colorbar(label='Log Frequency')  # Update the colorbar label)
    colorbar.set_ticks(np.log10(colorbar_ticks+0.01), labels=['$10^{-3}$', '$10^{-2}$', '$10^{-1}$'])  # Set custom tick locations on the color bar

    plt.xlabel('Layer')
    plt.ylabel('L2 Norm Value')
    plt.title('Norm Values Heatmap')
    plt.savefig(f'/mnt/imgnet/job/{epoch}_eval_norms_per_layer.png')



    if False:
        num_layers = len(norms)
        y_axis_max = 400
        # Create a heatmap array to store the frequency of values in each bin
        heatmap = np.zeros((y_axis_max, num_layers))

        # Calculate the histogram for each layer
        for layer in range(num_layers):
            # Flatten the norm values for the current layer
            flat_norms = norms[layer].flatten()
            
            # Calculate the 2D histogram
            hist, edges = np.histogram(flat_norms, bins=255, range=(0, 500))
            
            # Store the histogram in the heatmap
            heatmap[:,layer] = hist


        # Create the heatmap plot
        plt.imshow(heatmap, cmap='hot', origin='lower', aspect='auto', extent=[0, num_layers, 0, y_axis_max], interpolation='nearest')
        plt.colorbar(label='Frequency')
        plt.xlabel('Layers')
        plt.ylabel('Norm Values')
        plt.title('Norm Values Heatmap')
        plt.show()














    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
