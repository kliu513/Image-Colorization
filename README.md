# Image-Colorization

### Running the Model

1. Download [Places365](http://data.csail.mit.edu/places/places365/train_256_places365standard.tar) to the `Dataset` directory. 
2. Extract `places365_train_standard.txt` from `places365_train_standard.zip` to the `Dataset` directory. 
3. Start training the model by running all the blocks in `UNet_PatchGan_v4.ipnyb`. 
4. To evaluate the model with new pictures, modify `ckpt_path` and `img_path` in the third and last blocks of `Evaluate.ipnyb` respectively.

To change the root directory of datasets, modify `dataset.py` on Line 8 and 34. Be sure to include `places365_train_standard.txt` in the new directory.

### Things to Note

1. The model is mainly based on PatchGan. A classifier and Convolution Block Attention Module (CBAM) are incorporated to improve the model's performance.
2. For this model, pictures are represented in LAB color space. There are two main reasons why we utilize LAB instead of RGB: Given that the "L" channel can be used as input, the model only needs to generate and concatenate values in the "A" and "B" channels; on the other hand, it will be easier for the model to colorize images of varying size, since the rescaled "A" and "B" channels can be concatenated using the original "L" channel with no distortion in pictures.

### Link to the [Paper](https://github.com/kliu513/Image-Colorization/blob/94dda2c5c1f7565cc84189934a11fb8c82182c18/Paper.pdf)
