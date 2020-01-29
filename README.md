# CaptchaTests
Some starry-eyed neural nets trying to pass the Captcha test

A repository for some primitive attempts at a Captcha solver. The latest is a generative adversarial neural network,
whose output is to train a solver. This configuration is inspired by Ye et al. in their work, *Yet Another Text Captcha Solver:
A Generative Adversarial Network Based Approach*. Accordingly, the initial stage is simultaneously train a generator, which
attempts to reproduce authentic text Captchas from their string labels, and a discriminator, which attempts to discern between
authentic and generator-derived images. Currently, the generator employs two concepts from linear algebra that are not
commonplace in current machine learning methods. For this reason, the Captcha labels are converted into sparse diagonal matrices
of dimension equal to the length of the allowed Captcha character set. Each nonzero element of the matrix is 1 at the index
corresponding to its position in the character set. The first layer of the generator is a non-unitary basis transformation
operator, which rotates the sparse input into a dense form. The second layer is a sort of projection operator with an abritrary
number of output channels, each one mapping square dense matrix into a space with dimensions of the desired output images.
The tentative final layer is a 2D convolution with 3 channels for RGB images.

The discriminator features three 2D convolutions with inter-convolutional pooling. The final convolution is then flattened and
fed to a dense layer. The output is a 1-channel dense layer, for binary classification (i.e., authentic or synthetic).

