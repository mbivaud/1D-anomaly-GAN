from random import random

from numpy import ones, zeros

import dataset


def train_discriminator(model, n_epochs=1000, n_batch=128):
    half_batch = int(n_batch/2)
    # run epochs manually
    for i in range(n_epochs):
        # have real example
        x_real, y_real = dataset.get_real_samples()
        # update model
        model.train_on_batch(x_real, y_real)
        # generate fake examples
        x_fake = random.rand((n_batch, 91))
        y_fake = zeros((n_batch, 1))
        # update model
        model.train_on_batch(x_fake, y_fake)
        # evaluate the model
        _, acc_real = model.evaluate(x_real, y_real, verbose=0)
        _, acc_fake = model.evaluate(x_fake, y_fake, verbose=0)
        print(i, acc_real, acc_fake)


def train_gan(gan_model, latent_dim, n_epochs=1000, n_batch=128):
    # manually enumerate epochs
    for i in range(n_epochs):
        # prepare points in latent space as input for the generator
        x_gan = dataset.generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # update the generator via the discriminator's error
        gan_model.train_on_batch(x_gan, y_gan)
