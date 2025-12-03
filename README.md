# pointcloud-fl
This project reproduces Vlassis et al. (2023) for generating realistic 3D sand grains, but with the original DDPM replaced by flow matching. It includes a point cloud autoencoder and a UNet-based generative model reimplemented according to the original paper. Generation is performed in the latent space of the pre-trained autoencoder.

# Results: Point-Cloud Meshes

I provide a collection of generated latent vectors decoded back into 3D point-cloud meshes.  
You can view or download them here <https://drive.google.com/drive/folders/1MsTGhEDip9iwukcvqaRbYzt9-MpBKp8X?usp=sharing>.  

These examples were produced using the flow-matching model implemented in this repository.
