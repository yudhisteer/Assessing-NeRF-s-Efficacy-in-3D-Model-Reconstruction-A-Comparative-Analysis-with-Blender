# Neural Radiance Fields (NeRF) on custom synthetic datasets


## Plan of Action
1. Understanding NeRF
     - Volumetric Scene Representation
     - Volume Rendering
     - Improvement 1: Positional Encoding
     - Improvement 2: Hierarchical Volume Sampling

3. 3D Model using Blender

4. Training NeRF


---------------

## 1. Understanding NeRF






### 1.1 Volumetric Scene Representation
What has been done before NeRF is to have a set of images and use 3D CNN to predict a discrete volumetric representation such as a **Voxel Grid**. Though this technique has demonstrated impressive results, computing and storing these large voxel grids can  become computationally expensive for large and high-resolution scenes. What NeRF does is represent a scene as a **continuous** ```5D function``` which consists of **spatial 3D location** ```(x,y,z)``` of a point and the **2D viewing direction** ```(θ, φ)```. This is the **input**.

By using the 5D coordinates along camera rays as input, they can then represent any arbitrary scene as a **Fully Connected neural network (MLP)** - ```9 layers and 256 channels each```. By feeding those locations into the MLP, they produce the **emitted color** in ```(r,g,b)```  and the **volume density**, ```σ```. This is the **output**.



<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/673bcdde-e529-421c-ab15-fe0f8ff66475" width="70%" />
</p>
<div align="center">
    <p>Image Source: <a href="https://en.wikipedia.org/wiki/Spherical_coordinate_system">Spherical coordinate system</a></p>
</div>


Note: θ (theta) and φ (phi) represent the angular coordinates of a ray direction in **spherical coordinates** as shown below:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/06c54b78-3cc2-40fd-83ec-18527e3903f4" width="30%" />
</p>
<div align="center">
    <p>Image Source: <a href="https://en.wikipedia.org/wiki/Spherical_coordinate_system">Spherical coordinate system</a></p>
</div>

### 1.2 Volume Rendering



### 1.3 Improvement 1: Positional Encoding


### 1.4 Improvement 2: Hierarchical Volume Sampling



--------------------------






## References
1. https://www.youtube.com/watch?v=CRlN-cYFxTk&ab_channel=YannicKilcher
2. https://www.youtube.com/watch?v=LRAqeM8EjOo&ab_channel=BENMILDENHALL
3. https://www.fxguide.com/fxfeatured/the-art-of-nerfs-part1/?lid=7n16dhn58fxs
4. https://www.youtube.com/watch?v=CRlN-cYFxTk&t=1745s&ab_channel=YannicKilcher
5. https://www.fxguide.com/fxfeatured/the-art-of-nerfs-part-2-guassian-splats-rt-nerfs-stitching-and-adding-stable-diffusion-into-nerfs/
