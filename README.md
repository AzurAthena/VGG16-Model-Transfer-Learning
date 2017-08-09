# VGG16-Model-Transfer-Learning
## Transfer learning using tensorflow VGG16 image classification model

We have used the transfer learning VGG16 classification model to classify our own GOT dataset.
After creating trasnfer codes using the existing model, we have added fully connected and activation layers to perform a higher layer classfication on our GOT dataset.

The model reaches validation accuracy of ~95% in 20 Epochs and is extremely fast!

It shows how transfer learning can be used to solve classification tasks, by building sleek and prodigiously fast model.


Steps:
Run the model using file "transferModel.py"

Note:
Images transformations are created if number of images for any class are less than 500

References:
<br/>
Image transformation: https://github.com/vxy10/ImageAugmentation
<br/>
VGG16 Model: https://github.com/machrisaa/tensorflow-vgg
<br/>
Progress bar: https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html
