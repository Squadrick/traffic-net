A encoder-decoder style deep convolutional neural network for segmentation of traffic signs from dash camera video.

The network only uses around 170k weights, much lower than other modern networks.

It acheives this efficiency using bottleneck modules that downsamples the intermediate data tensors using 1-D convs.
