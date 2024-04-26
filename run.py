import os, argparse, subprocess, shlex, io, time, glob, pickle, pprint
# sh demo_deepseqslam.sh
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import tqdm, fire
from PIL import Image
Image.warnings.simplefilter('ignore')

import torch
import torch.nn as nn
import torch.utils.model_zoo
from torch.utils.data import Dataset, DataLoader
from pytorch_tcn import TCN
from netvlad import *
from hard_triplet_loss import *

import torchvision
import torchvision.models as models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

sequence_models = ['lstm', 'gru', 'tcn', 'transformer']

normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Gardens Point Training')
parser.add_argument('--data_path',
                    default='datasets/GardensPointWalking/',
                    help='path to dataset folder that contains preprocessed train and val *npy image files')
parser.add_argument('-o', '--output_path', default='checkpoints/',
                    help='path for storing model checkpoints')
parser.add_argument('--model_name', default='resnet18_lstm',
                    help='checkpoint model name (default: deepseqslam_resnet18_lstm)')
parser.add_argument('-a', '--cnn_arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--pretrained', default=True, type=bool,
                    help='use pre-trained CNN model (default: True)')
parser.add_argument('--val_set', default='day_right', type=str,
                    help='validation_set (default: day_right)')
parser.add_argument('--ngpus', default=2, type=int,
                    help='number of GPUs for training; 0 if you want to run on CPU (default: 2)')
parser.add_argument('-j', '--workers', default=1, type=int,
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--batch_size', default=32, type=int,
                    help='mini-batch size: 2^n (default: 32)')
parser.add_argument('--lr', '--learning_rate', default=.01, type=float,
                    help='initial learning rate (default: 1e-2)')
parser.add_argument('--load', default=False, type=bool,
                    help='restart training from last checkpoint')
parser.add_argument('--nimgs', default=200, type=int,
                    help='number of images (default: 200)')
parser.add_argument('--seq_len', default=10, type=int,
                    help='sequence length: ds (default: 10)')
parser.add_argument('--nclasses', default=196, type=int,
                    help='number of classes = nimgs - seq_len (default: 184)')
parser.add_argument('--img_size', default=128, type=int,
                    help='image size (default: 128)')
parser.add_argument('--sequence_model', metavar='MODEL', default='lstm',
                    choices=sequence_models,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: lstm)')

FLAGS, FIRE_FLAGS = parser.parse_known_args()

os.makedirs(f"results/{FLAGS.model_name.lower()}", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

def set_gpus(n=1):
    """
    Finds all GPUs on the system and restricts to n of them that have the most
    free memory.
    """
    gpus = subprocess.run(shlex.split(
        'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True, stdout=subprocess.PIPE).stdout
    gpus = pd.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
    gpus = gpus[gpus['memory.total [MiB]'] > 10000]  # only above 10 GB
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        visible = [int(i)
                   for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        gpus = gpus[gpus['index'].isin(visible)]
    gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # making sure GPUs are numbered the same way as in nvidia_smi
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        [str(i) for i in gpus['index'].iloc[:n]])

if FLAGS.ngpus > 0:
    set_gpus(FLAGS.ngpus)

def get_model(num_classes=FLAGS.nclasses):
    model = DeepSeqSLAM(num_classes)
    model = nn.DataParallel(model)

    if FLAGS.ngpus == 0:
        model = model.module  # remove DataParallel
    if FLAGS.ngpus > 0:
        model = model.cuda()
    return model

class DeepSeqSLAM(nn.Module):
    def __init__(self, num_classes):
        super(DeepSeqSLAM, self).__init__()

        if FLAGS.pretrained:
            print("=> Loading pre-trained model '{}'".format(FLAGS.cnn_arch))
            self.cnn = models.__dict__[FLAGS.cnn_arch](pretrained=FLAGS.pretrained)
            for param in self.cnn.parameters():
                param.requires_grad = False
        else:
            print("=> Using randomly inizialized model '{}'".format(FLAGS.cnn_arch))
            self.cnn = models.__dict__[FLAGS.cnn_arch](pretrained=FLAGS.pretrained)

        if FLAGS.cnn_arch == "resnet18":
            """ Resnet18 """
            self.feature_dim = 512
            #self.cnn.fc = nn.Identity()
            first_layers = list(self.cnn.children())[:-2]
            self.cnn = nn.Sequential(*first_layers)

        elif FLAGS.cnn_arch == "alexnet":
            """ Alexnet """
            self.feature_dim = 256
            #self.cnn.classifier[6] = nn.Identity()
            self.cnn = self.cnn.features

        elif FLAGS.cnn_arch == "vgg16":
            """ VGG16 """
            self.feature_dim = 512
            self.cnn = self.cnn.features
            #self.feature_dim = self.cnn.classifier[6].in_features
            #self.cnn.classifier[6] = nn.Identity()

        elif FLAGS.cnn_arch == "squeezenet1_0":
            """ Squeezenet """
            self.feature_dim = 512
            #self.cnn.classifier[1] = nn.Identity()
            self.cnn = self.cnn.features

        elif FLAGS.cnn_arch == "densenet161":
            """ Densenet """
            self.feature_dim = 2208
            #self.cnn.classifier = nn.Identity()
            self.cnn = self.cnn.features

        else:
            print("=> Please check model name or configure architecture for feature extraction only, exiting...")
            exit()

        self.num_classes = num_classes
        self.num_layers = 1
        self.input_size = self.feature_dim + 2
        self.hidden_units = 512
        self.dropout = 0.1
        
        self.netvlad = NetVLAD(num_clusters=1,dim=self.feature_dim)
        self.embednet = EmbedNet(self.cnn, self.netvlad)
        
        # sh demo_deepseqslam.sh
        if FLAGS.sequence_model == "lstm":
            self.sequence_model = nn.LSTM(self.input_size, self.hidden_units, self.num_layers, dropout=self.dropout, batch_first=True)

        elif FLAGS.sequence_model == "gru":
            self.sequence_model = nn.GRU(self.input_size, self.hidden_units, self.num_layers, dropout=self.dropout, batch_first=True)

        elif FLAGS.sequence_model == "tcn":
            self.sequence_model = TCN(self.input_size, [self.hidden_units], input_shape='NLC', dropout=self.dropout, 
                                      kernel_size=2, use_gate=True)

        elif FLAGS.sequence_model == "transformer":
            self.sequence_model = nn.Transformer(d_model=self.input_size, nhead=1, dropout=self.dropout,
                                                 num_encoder_layers=1, 
                                                 num_decoder_layers=1,  
                                                 norm_first=True, batch_first=True)

        else:
            print("=> Please check sequence model name or configure architecture for feature extraction only, exiting...")
            exit()

        if FLAGS.sequence_model == "transformer":
            self.mlp = nn.Linear(self.input_size, self.num_classes)
        else:
            self.mlp = nn.Linear(self.hidden_units, self.num_classes)

    def forward(self, inp):
        xs = inp[0].shape
        p = inp[1]
        x = inp[0]

	# Compute global descriptors
        x = x.view(xs[0]*xs[1],3,FLAGS.img_size,FLAGS.img_size) # 3xHXW
        x = self.embednet(x)
        #x = self.cnn(x)
        x = x.view(xs[0], xs[1], self.feature_dim)

        # Concatenate descriptor (x) with positional data (p)
        x = torch.cat((x,p),2)

	# Propagate through sequence model
        if FLAGS.sequence_model == "tcn":
            r_out = self.sequence_model(x)
        elif FLAGS.sequence_model == "transformer":
            r_out = self.sequence_model(x, x)
        else:
            r_out, _ = self.sequence_model(x, None)
        out = self.mlp(r_out[:,-1,:])
        return out

def train(restore_path=f'checkpoints/model_{FLAGS.model_name.lower()}.pth.tar',  
          save_train_epochs=.1,  
          save_val_epochs=.5,  
          save_model_epochs=5,  
          save_model_secs=60 * 1,  
          save_best_model=True,
          validate_every=5,
          patience=5):  

    print(FLAGS)

    trainer = DeepSeqSLAMTrain()

    print(trainer.model)
    
    best_val_loss = float('inf')  # Initialize best validation loss
    epochs_since_improvement = 0  # Initialize counter for epochs since last improvement

    start_epoch = 0
    if restore_path is not None and FLAGS.load:
        ckpt_data = torch.load(restore_path)
        start_epoch = ckpt_data['epoch']
        trainer.model.load_state_dict(ckpt_data['state_dict'])
        trainer.optimizer.load_state_dict(ckpt_data['optimizer'])

    records = []
    recent_time = time.time()

    nsteps = len(trainer.data_loader)
    if save_train_epochs is not None:
        save_train_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_train_epochs) * nsteps).astype(int)
    if save_val_epochs is not None:
        save_val_steps = (np.arange(0, FLAGS.epochs + 1,
                                    save_val_epochs) * nsteps).astype(int)
    if save_model_epochs is not None:
        save_model_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_model_epochs) * nsteps).astype(int)

    results = {'meta': {'step_in_epoch': 0,
                        'epoch': start_epoch,
                        'wall_time': time.time()}
               }
    global_acc = 0
    epoch_acc = 0
    
    epoch_train_losses = []  # List to store average training losses for each epoch
    epoch_val = []
    epoch_val_losses = []    # List to store validation losses for each epoch

    for epoch in tqdm.trange(start_epoch, FLAGS.epochs + 1, initial=start_epoch, desc='epoch'):
        data_load_start = np.nan
        loop = tqdm.tqdm(enumerate(trainer.data_loader), total=len(trainer.data_loader), leave=True)
        
        epoch_loss_sum = 0.0  # Accumulator for the training loss in the current epoch
        epoch_step_count = 0  # Counter for the number of steps in the current epoch

        for step, data in loop:
            data_load_time = time.time() - data_load_start
            global_step = epoch * len(trainer.data_loader) + step

            if FLAGS.output_path is not None:
                records.append(results)
                if len(results) > 1:
                    pickle.dump(records, open(os.path.join(FLAGS.output_path, f'results_{FLAGS.model_name.lower()}.pkl'), 'wb'))

                ckpt_data = {}
                ckpt_data['flags'] = FLAGS.__dict__.copy()
                ckpt_data['epoch'] = epoch
                ckpt_data['state_dict'] = trainer.model.state_dict()
                ckpt_data['optimizer'] = trainer.optimizer.state_dict()

                if save_model_secs is not None:
                    if time.time() - recent_time > save_model_secs:
                        torch.save(ckpt_data, os.path.join(FLAGS.output_path,
                                                           'latest_checkpoint.pth.tar'))
                        recent_time = time.time()

            else:
                if len(results) > 1:
                    pprint.pprint(results)

            if epoch < FLAGS.epochs:
                frac_epoch = (global_step + 1) / len(trainer.data_loader)
                record = trainer(frac_epoch, *data)
                record['data_load_dur'] = data_load_time
                results = {'meta': {'step_in_epoch': step + 1,
                                    'epoch': frac_epoch,
                                    'wall_time': time.time()}
                           }
                
                # Accumulate the training loss for the current epoch
                epoch_loss_sum += record['loss']
                epoch_step_count += 1

            loop.set_description(f"Epoch [{epoch}/{FLAGS.epochs + 1}]")
            loop.set_postfix(loss=record['loss'], top1=record['top1'], top5=record['top5'],lr=record['learning_rate'])

            data_load_start = time.time()
        
        # Calculate the average training loss for the current epoch
        if epoch_step_count > 0:
            epoch_avg_loss = epoch_loss_sum / epoch_step_count
            epoch_train_losses.append(epoch_avg_loss)

        trainer.lr.step(record['top1'])
        epoch_acc = record['top1']
        if save_best_model:
            if global_acc < epoch_acc or epoch == 0:
                global_acc = epoch_acc
                torch.save(ckpt_data, os.path.join(FLAGS.output_path,
                                                   f'model_{FLAGS.model_name.lower()}.pth.tar'))
        if epoch % validate_every == 0:
            epoch_val.append(epoch)
            val_loss = val(training=True)
            epoch_val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    print(f'Validation loss has not improved for {patience*validate_every} epochs. Early stopping...')
                    break  # Stop training
                    
    print('loss=', record['loss'], 'top1=', record['top1'], 'top5=', record['top5'], 'lr=', record['learning_rate'])

    # Plot and save training loss curve
    # Plot both training and validation loss curves
    plt.plot(epoch_train_losses, label='Training Loss')
    plt.plot(epoch_val, epoch_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.savefig(f'results/{FLAGS.model_name.lower()}/training_and_validation_loss_curves.jpg', bbox_inches='tight')

class SequentialDataset(Dataset):
    """Sequence-based dataset."""
    
    def __init__(self, csv_file, data_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with positional data normalized between 0 to 1.
            data_dir (string): Directory with all the images of a route.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = FLAGS.img_size
        self.total_imgs = np.sort(os.listdir(data_dir))
        self.ids = np.linspace(0,FLAGS.nimgs-1,FLAGS.nimgs)
        self.pos = 1000*((np.loadtxt(csv_file, delimiter=',')-0.5)*2)
        
    def __len__(self):
        return len(self.total_imgs) - FLAGS.seq_len

    def __getitem__(self, idx):

        img_seq = []
        pos_seq = self.pos[idx:idx+FLAGS.seq_len]
        for i in range(FLAGS.seq_len):
            img_loc = os.path.join(self.data_dir, self.total_imgs[idx+i])
            img_seq += [Image.open(img_loc)]

        ids = self.ids[idx]
        ids = np.array(ids)

        img_seq_pt = []
        if self.transform:
            for images in img_seq:
                img_seq_pt += [torch.unsqueeze(self.transform(images), 0)]

        img_seq = torch.cat(img_seq_pt, dim=0)
        ids = torch.from_numpy(ids).type(torch.long)
        pos_seq = torch.from_numpy(pos_seq.astype('float32'))

        return (img_seq, pos_seq), ids

class DeepSeqSLAMTrain(object):

    def __init__(self):

        self.name = 'train'
        self.data_loader = self.data()
        num_classes = FLAGS.nimgs - FLAGS.seq_len
        self.model = get_model(num_classes)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=FLAGS.lr)
        self.lr = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1,
                      patience=15, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-6,
                      eps=1e-08, verbose=False)

        self.loss = nn.CrossEntropyLoss()
        #self.loss = HardTripletLoss()
        if FLAGS.ngpus > 0:
            self.loss = self.loss.cuda()

    def data(self):
        dataset = SequentialDataset(csv_file=os.path.join(FLAGS.data_path, f'gp_pos.csv'),
                                   data_dir=os.path.join(FLAGS.data_path, f'{FLAGS.val_set}'),
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.Resize((FLAGS.img_size, FLAGS.img_size)),
                                       torchvision.transforms.ToTensor(),
                                       normalize,
                                   ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=False, num_workers=FLAGS.workers, pin_memory=True)
        return data_loader

    def __call__(self, frac_epoch, inp, target):
        start = time.time()
        if FLAGS.ngpus > 0:
            target = target.cuda(non_blocking=True)
        output = self.model(inp)
        record = {}
        loss = self.loss(output, target)
        record['loss'] = loss.item()
        record['top1'], record['top5'] = accuracy(output, target, topk=(1, 5))
        record['top1'] /= len(output)
        record['top5'] /= len(output)
        record['learning_rate'] = self.optimizer.param_groups[0]['lr']

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        record['dur'] = time.time() - start
        return record


def val(restore_path=f'checkpoints/model_{FLAGS.model_name.lower()}.pth.tar', training=False):

    validator = DeepSeqSLAMVal(training)
    model = validator.model

    if restore_path is not None:
        ckpt_data = torch.load(restore_path)
        start_epoch = ckpt_data['epoch']
        model.load_state_dict(ckpt_data['state_dict'], strict=False)
        print('Model restored!')

    results = validator()

    print('loss = ', results['loss'], 'top1 = ', results['top1'],
          'top5 = ', results['top5'])

    return results['loss']

class DeepSeqSLAMVal(object):

    def __init__(self, training):
        self.name = 'val'
        self.training = training
        
        self.data_loader = self.data()
        num_classes = FLAGS.nimgs - FLAGS.seq_len
        self.model = get_model(num_classes)

        self.loss = nn.CrossEntropyLoss(size_average=False)
        #self.loss = HardTripletLoss()
        if FLAGS.ngpus > 0:
            self.loss = self.loss.cuda()

    def data(self):
        dataset = SequentialDataset(csv_file=os.path.join(FLAGS.data_path, f'gp_pos.csv'),
                                   data_dir=os.path.join(FLAGS.data_path, f'{FLAGS.val_set}'),
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.Resize((FLAGS.img_size, FLAGS.img_size)),
                                       torchvision.transforms.ToTensor(),
                                       normalize]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=False, num_workers=FLAGS.workers, pin_memory=True)
        return data_loader

    def __call__(self):
        self.model.eval()
        record = {'loss': 0, 'top1': 0, 'top5': 0}
        samples = 0
        with torch.no_grad():
            y_pred = []
            sim_m = []
            for (inp, target) in tqdm.tqdm(self.data_loader, desc=self.name):

                if FLAGS.ngpus > 0:
                    target = target.cuda(non_blocking=True)
                output = self.model(inp)
                _, predicted = torch.max(output.data, 1)
                sim_m.append(output)
                y_pred.append(predicted)
                samples += len(output)
                record['loss'] += self.loss(output, target).item()
                p1, p5 = accuracy(output, target, topk=(1, 5))
                record['top1'] += p1
                record['top5'] += p5

        for key in record:
            record[key] /= samples

        if not self.training:
            y_pred = torch.cat(y_pred, dim=0).data.cpu().numpy()

            plt.figure()
            plt.plot(y_pred,',')
            plt.savefig(f'results/{FLAGS.model_name.lower()}/best_matches_{FLAGS.val_set}.jpg', bbox_inches='tight')
            
            sim_m = torch.cat(sim_m, dim=0).data.cpu().numpy()
            plt.imshow(sim_m,cmap='jet_r')
            plt.colorbar()
            plt.savefig(f'results/{FLAGS.model_name.lower()}/diff_matrix_{FLAGS.val_set}.jpg', bbox_inches='tight')

        return record


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res


if __name__ == '__main__':
    fire.Fire(command=FIRE_FLAGS)
