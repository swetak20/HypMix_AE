import manifolds.poincare.math as pmath_geo
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
pl.seed_everything(42)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

DATASET_PATH = "../data"
CHECKPOINT_PATH = "../saved_models/tutorial9"

# Transformations applied on each image => only make them a tensor
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,))])

# Loading the training dataset. We need to split it into a training and validation part
train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=transform, download=True)
pl.seed_everything(42)
train_set, val_set = torch.utils.data.random_split(train_dataset, [45000, 5000])

# Loading the test set
test_set = CIFAR10(root=DATASET_PATH, train=False, transform=transform, download=True)

# We define a set of data loaders that we can use for various purposes later.
train_loader = data.DataLoader(train_set, batch_size=256, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)
test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)

def get_train_images(num):
    return torch.stack([train_dataset[i][0] for i in range(num)], dim=0)

class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*16*c_hid, 2*16),
            act_fn(),
            nn.Linear(2*16, 8),
            act_fn(),
            nn.Linear(8,2)
        )

    def forward(self, x):
        out = self.net(x)
        return out

class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2*16*c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        
        return x

class Autoencoder(pl.LightningModule):

    def __init__(self,
                 base_channel_size: int,
                 latent_dim: int,
                 encoder_class : object = Encoder,
                 decoder_class : object = Decoder,
                 num_input_channels: int = 3,
                 width: int = 32,
                 height: int = 32):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, _ = batch # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=20,
                                                         min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)

class GenerateCallback(pl.Callback):

    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1,1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)

def train_cifar(latent_dim):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"cifar10_{384}"),
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=5,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    GenerateCallback(get_train_images(8), every_n_epochs=10),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"cifar10_{latent_dim}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = Autoencoder.load_from_checkpoint(pretrained_filename)
    else:
        model = Autoencoder(base_channel_size=32, latent_dim=latent_dim)
        trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result


model_ld, result_ld = train_cifar(384)


def HypMix(x1, x2, l):
	x1 = pmath_geo.expmap0(x1, c=torch.tensor([1.0], device='cuda'))
	x2 = pmath_geo.expmap0(x2, c=torch.tensor([1.0], device='cuda'))
	xx1 = pmath_geo.mobius_scalar_mul(l, x1, c=torch.tensor([1.0], device='cuda'))
	xx2 = pmath_geo.mobius_scalar_mul((1-l), x2, c=torch.tensor([1.0], device='cuda'))
	y2 = pmath_geo.mobius_add(xx1, xx2, c=torch.tensor([1.0], device='cuda'))
	z = pmath_geo.logmap0(y2, c=torch.tensor([1.0], device='cuda'))
	return z

'''Latent Space representation obtained from autoencoders to feed into HypMix'''
t = torch.tensor([[-0.0172,  0.0105],
		        [-0.0172,  0.0113],
		        [-0.0177,  0.0104],
		        [-0.0170,  0.0106],
		        [-0.0177,  0.0111],
		        [-0.0175,  0.0108],
		        [-0.0175,  0.0108],
		        [-0.0179,  0.0105],
		        [-0.0179,  0.0108],
		        [-0.0171,  0.0109],
		        [-0.0170,  0.0099],
		        [-0.0177,  0.0108],
		        [-0.0165,  0.0112],
		        [-0.0173,  0.0111],
		        [-0.0172,  0.0114],
		        [-0.0173,  0.0111],
		        [-0.0177,  0.0115],
		        [-0.0171,  0.0109],
		        [-0.0181,  0.0111],
		        [-0.0173,  0.0109],
		        [-0.0175,  0.0113],
		        [-0.0176,  0.0121],
		        [-0.0176,  0.0111],
		        [-0.0165,  0.0115],
		        [-0.0175,  0.0105],
		        [-0.0171,  0.0108],
		        [-0.0171,  0.0110],
		        [-0.0174,  0.0119],
		        [-0.0173,  0.0110],
		        [-0.0173,  0.0108],
		        [-0.0172,  0.0100],
		        [-0.0168,  0.0104],
		        [-0.0181,  0.0114],
		        [-0.0180,  0.0115],
		        [-0.0174,  0.0115],
		        [-0.0174,  0.0109],
		        [-0.0181,  0.0110],
		        [-0.0174,  0.0105],
		        [-0.0171,  0.0114],
		        [-0.0174,  0.0109],
		        [-0.0171,  0.0115],
		        [-0.0166,  0.0113],
		        [-0.0168,  0.0113],
		        [-0.0174,  0.0112],
		        [-0.0171,  0.0116],
		        [-0.0176,  0.0103],
		        [-0.0171,  0.0112],
		        [-0.0176,  0.0118],
		        [-0.0178,  0.0110],
		        [-0.0168,  0.0115],
		        [-0.0172,  0.0103],
		        [-0.0174,  0.0109],
		        [-0.0180,  0.0113],
		        [-0.0177,  0.0116],
		        [-0.0173,  0.0107],
		        [-0.0176,  0.0110],
		        [-0.0174,  0.0112],
		        [-0.0174,  0.0109],
		        [-0.0171,  0.0111],
		        [-0.0169,  0.0113],
		        [-0.0168,  0.0103],
		        [-0.0176,  0.0115],
		        [-0.0171,  0.0107],
		        [-0.0173,  0.0112],
		        [-0.0174,  0.0116],
		        [-0.0175,  0.0109],
		        [-0.0171,  0.0108],
		        [-0.0174,  0.0116],
		        [-0.0177,  0.0113],
		        [-0.0169,  0.0109],
		        [-0.0183,  0.0110],
		        [-0.0173,  0.0110],
		        [-0.0178,  0.0117],
		        [-0.0177,  0.0114],
		        [-0.0165,  0.0112],
		        [-0.0173,  0.0115],
		        [-0.0179,  0.0109],
		        [-0.0175,  0.0110],
		        [-0.0174,  0.0109],
		        [-0.0173,  0.0116],
		        [-0.0174,  0.0122],
		        [-0.0175,  0.0107],
		        [-0.0180,  0.0110],
		        [-0.0178,  0.0109],
		        [-0.0180,  0.0113],
		        [-0.0172,  0.0108],
		        [-0.0173,  0.0109],
		        [-0.0171,  0.0105],
		        [-0.0176,  0.0113],
		        [-0.0167,  0.0112],
		        [-0.0175,  0.0109],
		        [-0.0168,  0.0108],
		        [-0.0177,  0.0114],
		        [-0.0173,  0.0107],
		        [-0.0181,  0.0109],
		        [-0.0172,  0.0111],
		        [-0.0169,  0.0109],
		        [-0.0168,  0.0117],
		        [-0.0181,  0.0118],
		        [-0.0175,  0.0111],
		        [-0.0175,  0.0118],
		        [-0.0179,  0.0109],
		        [-0.0171,  0.0122],
		        [-0.0174,  0.0109],
		        [-0.0174,  0.0119],
		        [-0.0177,  0.0112],
		        [-0.0182,  0.0106],
		        [-0.0175,  0.0111],
		        [-0.0173,  0.0107],
		        [-0.0171,  0.0108],
		        [-0.0170,  0.0109],
		        [-0.0174,  0.0108],
		        [-0.0172,  0.0106],
		        [-0.0172,  0.0114],
		        [-0.0173,  0.0120],
		        [-0.0174,  0.0108],
		        [-0.0173,  0.0108],
		        [-0.0173,  0.0113],
		        [-0.0177,  0.0104],
		        [-0.0176,  0.0109],
		        [-0.0174,  0.0108],
		        [-0.0171,  0.0108],
		        [-0.0187,  0.0112],
		        [-0.0181,  0.0117],
		        [-0.0172,  0.0112],
		        [-0.0177,  0.0112],
		        [-0.0178,  0.0118],
		        [-0.0168,  0.0114],
		        [-0.0175,  0.0112],
		        [-0.0170,  0.0110],
		        [-0.0176,  0.0110],
		        [-0.0174,  0.0114],
		        [-0.0175,  0.0114],
		        [-0.0176,  0.0098],
		        [-0.0172,  0.0108],
		        [-0.0169,  0.0108],
		        [-0.0176,  0.0109],
		        [-0.0179,  0.0110],
		        [-0.0178,  0.0111],
		        [-0.0171,  0.0111],
		        [-0.0178,  0.0111],
		        [-0.0173,  0.0112],
		        [-0.0170,  0.0104],
		        [-0.0170,  0.0113],
		        [-0.0173,  0.0110],
		        [-0.0172,  0.0112],
		        [-0.0173,  0.0117],
		        [-0.0163,  0.0113],
		        [-0.0171,  0.0110],
		        [-0.0174,  0.0109],
		        [-0.0172,  0.0109],
		        [-0.0178,  0.0111],
		        [-0.0176,  0.0106],
		        [-0.0178,  0.0108],
		        [-0.0173,  0.0112],
		        [-0.0174,  0.0112],
		        [-0.0179,  0.0120],
		        [-0.0168,  0.0104],
		        [-0.0172,  0.0106],
		        [-0.0169,  0.0118],
		        [-0.0174,  0.0107],
		        [-0.0170,  0.0120],
		        [-0.0165,  0.0108],
		        [-0.0170,  0.0115],
		        [-0.0175,  0.0106],
		        [-0.0173,  0.0113],
		        [-0.0174,  0.0111],
		        [-0.0176,  0.0110],
		        [-0.0172,  0.0111],
		        [-0.0175,  0.0111],
		        [-0.0178,  0.0117],
		        [-0.0172,  0.0115],
		        [-0.0175,  0.0111],
		        [-0.0168,  0.0106],
		        [-0.0171,  0.0107],
		        [-0.0174,  0.0101],
		        [-0.0174,  0.0111],
		        [-0.0178,  0.0111],
		        [-0.0171,  0.0113],
		        [-0.0170,  0.0108],
		        [-0.0174,  0.0115],
		        [-0.0164,  0.0115],
		        [-0.0173,  0.0113],
		        [-0.0180,  0.0111],
		        [-0.0173,  0.0105],
		        [-0.0178,  0.0104],
		        [-0.0175,  0.0108],
		        [-0.0177,  0.0116],
		        [-0.0178,  0.0111],
		        [-0.0189,  0.0115],
		        [-0.0178,  0.0110],
		        [-0.0174,  0.0111],
		        [-0.0173,  0.0108],
		        [-0.0177,  0.0110],
		        [-0.0182,  0.0108],
		        [-0.0174,  0.0110],
		        [-0.0175,  0.0107],
		        [-0.0176,  0.0108],
		        [-0.0177,  0.0115],
		        [-0.0172,  0.0115],
		        [-0.0168,  0.0100],
		        [-0.0178,  0.0111],
		        [-0.0180,  0.0106],
		        [-0.0186,  0.0102],
		        [-0.0174,  0.0109],
		        [-0.0177,  0.0105],
		        [-0.0177,  0.0116],
		        [-0.0172,  0.0111],
		        [-0.0169,  0.0114],
		        [-0.0170,  0.0109],
		        [-0.0169,  0.0112],
		        [-0.0176,  0.0109],
		        [-0.0178,  0.0106],
		        [-0.0173,  0.0116],
		        [-0.0174,  0.0117],
		        [-0.0173,  0.0112],
		        [-0.0182,  0.0104],
		        [-0.0176,  0.0109],
		        [-0.0175,  0.0115],
		        [-0.0173,  0.0112],
		        [-0.0172,  0.0111],
		        [-0.0169,  0.0116],
		        [-0.0169,  0.0111],
		        [-0.0166,  0.0117],
		        [-0.0182,  0.0112],
		        [-0.0169,  0.0112],
		        [-0.0169,  0.0104],
		        [-0.0171,  0.0107],
		        [-0.0181,  0.0107],
		        [-0.0170,  0.0110],
		        [-0.0178,  0.0110],
		        [-0.0170,  0.0105],
		        [-0.0186,  0.0110],
		        [-0.0182,  0.0114],
		        [-0.0179,  0.0107],
		        [-0.0175,  0.0111],
		        [-0.0169,  0.0113],
		        [-0.0179,  0.0113],
		        [-0.0172,  0.0111],
		        [-0.0170,  0.0107],
		        [-0.0171,  0.0104],
		        [-0.0177,  0.0107],
		        [-0.0176,  0.0120],
		        [-0.0169,  0.0105],
		        [-0.0179,  0.0112],
		        [-0.0172,  0.0110],
		        [-0.0177,  0.0114],
		        [-0.0180,  0.0110],
		        [-0.0178,  0.0110],
		        [-0.0175,  0.0108],
		        [-0.0172,  0.0105],
		        [-0.0172,  0.0111],
		        [-0.0176,  0.0107],
		        [-0.0167,  0.0110],
		        [-0.0175,  0.0111],
		        [-0.0166,  0.0110]])


