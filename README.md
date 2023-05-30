# GANs-Model
The aim of this project is to explore the capabilities of Generative Adversarial Networks (GANs) in the generation of anime faces.
 GANs are a generative modeling approach that uses deep learning techniques, such as convolutional neural networks, to create new, realistic and varied examples.
In this project, we developed a generative model using a pre-existing anime dataset. The generator takes as input a latent tensor, usually represented by a vector or matrix of random numbers, and transforms it into an image representing an anime face. To achieve this goal, we used transposed convolution layers, also known as inverse convolutions, to improve the quality and detail of the generated images.

Using Gradio, a user-friendly library for creating user interfaces, we developed an interactive interface for generating and displaying the anime faces produced by our model in real time. You can interact with this interface to generate images and view them instantly.

In the context of deploying an Anime Faces Generator template, we used Flask to create a user-friendly web interface where users can interact with the template to generate anime images.

The dataset used : https://www.kaggle.com/datasets/splcher/animefacedataset
