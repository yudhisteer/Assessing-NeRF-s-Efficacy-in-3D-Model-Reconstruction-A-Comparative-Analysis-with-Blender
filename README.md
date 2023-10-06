# Neural Radiance Fields (NeRF) on custom synthetic datasets


## Plan of Action

0. Prerequisites
      - Ray Tracing
      - Ray Casting
      - Ray Marching
      - NeRFing a sphere: Part I

1. Understanding NeRF
     - Volumetric Scene Representation
     - Volume Rendering
     - NeRFing a sphere: Part II
     - Improvement 1: Positional Encoding
     - Improvement 2: Hierarchical Volume Sampling
     - NeRFing a sphere: Part II

2. 3D Model using Blender

3. Training NeRF


---------------
## 0. Prerequisites

### 0.1 Ray Tracing


### 0.2 Ray Casting

### 0.3 Ray Marching

NeRF (Neural Radiance Fields) uses a concept similar to ray casting but differs in some key aspects. Here's how NeRF works:

Input: NeRF takes 2D images captured from various viewpoints as its input. These images provide information about how the 3D scene appears from different angles.

Ray Casting: NeRF performs a form of ray casting internally. For each pixel in a 2D image, it casts rays into the 3D scene, originating from the camera's viewpoint and passing through that pixel's location on the image.

Neural Network: Instead of directly calculating intersections and shading like traditional ray casting, NeRF uses a neural network to learn the 3D representation of the scene. The neural network takes the rays' directions and origins as input and predicts the 3D scene's appearance and structure at those points.

Scene Reconstruction: NeRF leverages the predictions of the neural network to reconstruct the 3D scene. It learns the scene's geometry (shape) and appearance (color and texture) by training on the set of images.

So, while NeRF does involve a form of ray casting where rays are cast from the camera into the scene, it differs from traditional ray casting in that it relies on a neural network to learn and reconstruct the 3D scene's properties. It's more closely related to neural networks and 3D scene representation rather than the classic ray tracing or simple ray casting techniques used for rendering or visibility testing.

Pixel in a 2D Image: Consider a 2D image, like a photograph or a frame from a video. Each pixel in this image corresponds to a point in the 3D space (the scene you're trying to model).

Casts Rays: NeRF virtually casts rays from the camera's viewpoint through each pixel in the 2D image. Each ray represents a line in 3D space, originating at the camera's position and passing through a specific pixel's location on the image.

Into the 3D Scene: These rays extend into the 3D scene, essentially probing the scene's geometry and appearance. The purpose of casting rays from each pixel is to collect information about how light interacts with the scene, which helps NeRF build a 3D representation of the scene. The algorithm uses this information to reconstruct the 3D scene's geometry and appearance. By gathering data from multiple rays cast from different pixels in the image, NeRF can create a more accurate and detailed 3D model.  the phrase "casts rays into the 3D scene" means that NeRF sends out virtual rays from the camera's viewpoint through each pixel in the 2D image to understand and model the 3D scene.


### 0.4 NeRFing a sphere
In this section, we will apply ray-casting techniques in order to create a sphere. We start by defining the difference between a **line** and a **ray**. while a line is an infinite straight path extending in **both directions**, a ray has an **origin** and a **direction vector** that extends infinitely in **one** direction. The equation of a ray can be modeled as a parametric equation:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/56095af0-764d-431b-8361-3d46a55947de"/>
</p>

where ```o``` is the **origin**, ```d``` is the **direction vector**, ```t``` is a **parameter** that varies along the ray and determines different points along its path, and ```r``` is a **position vector** representing any point on the ray.

Let's look at an example of how we can apply this equation. Suppose we have the origin of a vector at (2,3) with a direction vector of (1,1). We want to know the position vector of that ray when t=5. Using the equation above:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/f8ad3b8d-3755-4600-9268-25f025f2c244"/>
</p>

Note that (7,8) is the horizontal and vertical displacement in the x and y directions. That is we have moved 7.07 units along the ray, using Pythagoras theorem, from point A to point C and not 5 units as specified by t. If we want to move 5 units along that ray, then we need to normalize our direction vector into a unit vector. 

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/6a0529b3-5c7b-4366-9b91-84d0358605af"/>
</p>

We then re-calculate our position vector which is now (5.5, 6.5). If we check again with Pythagoras theorem, then we have indeed moved 5 units along that ray.

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/e477a759-6aa6-4a40-8505-8a297c87557b"/>
</p>

<table>
  <tr>
    <td><img width="379" alt="image" src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/f2e93073-92bd-46bd-87b7-a58c38e33290"></td>
    <td><img width="358" alt="image" src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/83a9ce1d-1618-4ced-a1d8-785e94bdf0f3"></td>
    <td><img width="360" alt="image" src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/f5ba7de3-a43a-40b3-91f6-4572d8b872db"></td>
  </tr>
</table>


We will first work on the mathematical calculations of how we can model a circle. Since it is easier to work in 2D, when modeling for a 3D sphere we will just need to add a ```z``` component. We will change our equation of a ray with different variables to avoid any notation confusion in the future. Note that I also separated it into their ```x-y``` components:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/47eff1bc-4cd9-4ded-aa9a-7fbd8ac387f6"/>
</p>

Below is the equation of a circle where a and b are the center and r is the radius. 

<p align="center">
  <img src=https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/8985882e-d998-4d8f-b8b6-35059854216b/>
</p>

Suppose we have a circle centered at the origin with radius 3. We also have a ray with an origin (-4,4) with a direction vector of (-1,-1), we want to know if that ray will intersect with the circle, and if so, where? 

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/bb614489-0db0-4e83-be8c-df8fae709e22" width="40%"/>
</p>





Our logic will be as follows:



```python
if intersection:
      pixel_color = "red"
else:
      pixel_color = "black" #background color
```

We start by replacing the x and y components of our ray equation into the equation of the sphere:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/80b77287-c213-46ac-9797-6c09233de40c"/>
</p>

We now expand the equation and remove t outside the bracket:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/39e69910-3faa-4373-ae98-4b5ed7e9547e"/>
</p>

In order to solve this quadratic equation, we can use the quadratic formula:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/01d96934-b062-4797-a5ca-2f0abcdb70cc"/>
</p>

with the discriminant being: 

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/309a646d-c9ed-4c7b-97bc-4786a0e9fed6"/>
</p>

where:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/07562b4e-924f-4110-9b21-2c93b54052e0"/>
</p>

Note that we can first check if we have any solution at all by plugging in the values into the discriminant. Normally, if the discriminant = 0, then we have one solution such that the line is tangent to the circle, if the discriminant > 0, then we have 2 solutions with the line intersecting the circle at two distinct positions and finally, if the discriminant < 0, then we have 0 solutions, with the line not intersecting the circle at all. Below is a graphical representation of it:


<table>
  <tr>
    <th align="center"> 1 Solution</th>
    <th align="center"> 2 Solutions</th>
    <th align="center"> 0 Solution</th>
  </tr>
  <tr>
    <td><img width="500" alt="Image 1" src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/eeaa27c6-f768-4db5-9abf-6031cc819c05"></td>
    <td><img width="500" alt="Image 2" src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/27d34962-27b2-4660-ad4c-7f18a73f4a57"></td>
    <td><img width="454" alt="Image 3" src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/60910fa6-a969-480b-b621-f193f07f500a"></td>
  </tr>
</table>

We then solve for t and plug the values of the latter into our equation for the ray where we get the x and y values for the point of intersections which are: (-2,12, 2.12) and (2.12, -2.12). Now let's implement it in python but for s sphere. Our quadratic formula will change to:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/dbb7704b-9e65-4655-9a54-2be339477ece"/>
</p>

We start by creating a class for the sphere whereby we will first compute the discriminant and then check if the latter >= 0, then we will color the pixel yellow. Note that previously, we already have created our rays which will originate from (0,0,0) and project downwards the z-axis. Our goal will be similar to that of the circle above, find where the rays intersect and then color the pixel.

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/3df8756d-ac8e-498f-8cba-5654d19458e8" width="50%"/>
</p>

Note that previously, we set the origin as the circle at (0,0), here we will incorporate the center as a variable which can be changed.

```python
class Sphere ():

    def __init__(self, center, radius, color):
        self.center = center
        self.radius = radius
        self.color = color

    def intersect (self, ray_origin, ray_direction):

        # we want to solve:
        # (bx^2 + by^2 + bz^2)t^2 + 2(axbx + ayby + azbz)t + (ax^2 + ay^2 + az^2 - r^2) = 0
        # where:
        # a = ray origin
        # b = ray direction
        # r = circle radius
        # t = hit distance

        # Center components
        cx = self.center[0]
        cy = self.center[1]
        cz = self.center[2]

        # Ray direction components
        bx = ray_direction[:, 0] #(160000,)
        by = ray_direction[:, 1]
        bz = ray_direction[:, 2]


        # Ray origin components
        ax = ray_origin[:, 0]
        ay = ray_origin[:, 1]
        az = ray_origin[:, 2]

        a = bx**2 + by**2 + bz**2
        b = 2 * (((ax-cx)*bx) + ((ay-cy)*by) + ((az-cz)*bz))
        c = (ax-cx)**2 + (ay-cy)**2 + (az-cz)**2 - self.radius**2

        # Store colors for each ray.
        intersection_points = []
        num_rays = ray_origin.shape[0] #16000
        colors = np.zeros((num_rays, 3))
        #intersection_points = np.zeros((num_rays, 3))

        ## Quadratic formula discriminant: b^2 - 4ac
        discriminant = b**2 - 4 * a * c #(160000,)

        # Iterate through the rays and check for intersection.
        for i in range(num_rays):
            if discriminant[i] >= 0:
                # Calculate the intersection point (quadratic formula)
                t1 = (-b[i] + np.sqrt(discriminant[i])) / (2 * a[i])
                t2 = (-b[i] - np.sqrt(discriminant[i])) / (2 * a[i])
                # Calculate both intersection points (plug in ray equation)
                intersection_point1 = ray_origin[i] + t1 * ray_direction[i]
                intersection_point2 = ray_origin[i] + t2 * ray_direction[i]

                # Store both intersection
                intersection_points.append([intersection_point1, intersection_point2])

                # Assign the sphere's color to rays that intersect the sphere.
                colors[i] = self.color

        return intersection_points, colors
```
Here's the result:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/8fc823ee-7deb-4d87-bcec-281331a4591a" width="45%"/>
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/f1d1d661-34eb-4155-a9f1-a76b332b7ebd" width="45%">
</p>





--------------------------

## 1. Understanding NeRF






### 1.1 Volumetric Scene Representation
What has been done before NeRF is to have a set of images and use 3D CNN to predict a discrete volumetric representation such as a **Voxel Grid**. Though this technique has demonstrated impressive results, however, computing and storing these large voxel grids can  become computationally expensive for large and high-resolution scenes. What NeRF does is represent a scene as a **continuous** ```5D function``` which consists of **spatial 3D location** ```x = (x,y,z)``` of a point and the **2D viewing direction** ```d = (θ, φ)```. This is the **input**.

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/29a33610-e1d3-4800-a1eb-2e2f01777cf0" width="70%" />
</p>


By using the 5D coordinates along camera rays as input, they can then represent any arbitrary scene as a **Fully Connected neural network (MLP)** - ```9 layers and 256 channels each```. By feeding those locations into the MLP, they produce the **emitted color** in ```c = (r,g,b)```  and the **volume density**, ```σ```. This is the **output**.

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/895e4a50-006c-452f-a3c1-f1fbaf610cd2" />
</p>

From the function above, we want to optimize the **weights** ![CodeCogsEqn (4)](https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/c0bcb685-e5b1-4ffb-af58-5060316a7453) that effectively associates each 5D coordinate input with its respective volume density and emitted directional color that represents the radiance.


<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/673bcdde-e529-421c-ab15-fe0f8ff66475" width="70%" />
</p>
<div align="center">
    <p>Image Source: <a href="https://en.wikipedia.org/wiki/Spherical_coordinate_system">Spherical coordinate system</a></p>
</div>

Let's explain the overall pipeline of the NeRF architecture:

**a)** Images are generated by selecting specific 5D coordinates that include both spatial location and the direction in which the camera is looking, all along the paths of camera rays.

**b)** Using these positions, we input them into a Multi-Layer Perceptron (MLP) to generate both color and volume density information.

**c)** We apply volume rendering techniques to combine these values into a final image.

**d)** Since this rendering function is differentiable, we can improve our scene representation by minimizing the difference between the images we synthesize and the ground truth images we've observed

Moreover, the author argues that they promote multiview consistency in the representation by constraining the network to estimate the **volume density**, σ as a function of the s**patial position** (**x**) exclusively. At the same time, they enable the prediction of RGB color (**c**) as a function of both the **spatial position** (**x**) **and** the **viewing direction (d)**.


**Note:**

1. θ (theta) and φ (phi) represent the angular coordinates of a ray direction in **spherical coordinates** as shown below.
2. The output density represents the probability distribution of how much of a 3D point in the scene is occupied by the object or scene surface. More precisely, it indicates whether a particular 3D point along a viewing ray intersects with the object's surface or not.

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/06c54b78-3cc2-40fd-83ec-18527e3903f4" width="30%" />
</p>
<div align="center">
    <p>Image Source: <a href="https://en.wikipedia.org/wiki/Spherical_coordinate_system">Spherical coordinate system</a></p>
</div>







### 1.2 Volume Rendering
Recall that density, σ, can be binary, where it equals ```1``` if the point is on the object's surface, i.e., it intersects with the scene geometry, and ```0``` if it is in empty space. Hence, everywhere in space, there is a value that represents density and color at that point in space.

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/5e70e0bc-ac28-4816-b27f-f53b0d97b501" width="60%" />
</p>
<div align="center">
    <p>Image Source: <a href="https://en.wikipedia.org/wiki/Spherical_coordinate_system">Spherical coordinate system</a></p>
</div>


We start by shooting a ray (camera ray) in our scene as shown below by Ray 1 and Ray 2. The equation of the camera ray is dependent on the origin, **o**, and the viewing direction, **d** for different time t.

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/5aa4118c-f427-4a19-a3c2-de52d4a3c30e" width="10%" />
</p>

We then sample a few points along the ray. For each point, we record the density and color at this point in space. We calculate the expected color as such:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/d7aa1eb3-722f-4ec2-83f9-26e18cde34b8" width="30%" />
</p>

- From the equation above, we observe we have the product of the density at point **r**(t): ```(σ(r(t)))``` which is independent of viewing direction **d** and the color at point **r**(t) from viewing direction **d**: ```(c(r(t),d))```. This means that if the density is 0, the color has no impact. But if we have a high density, the color has a bigger weight.

- We also have the term ```T(t)``` which is defined as the ```accumulated transmittance```. This refers to how much light is transmitted or attenuated along a viewing ray as it passes through the scene. So basically we will compute the density accumulated. Consider a scenario where there are two objects, A and B, positioned such that A is situated behind B. In this arrangement, A becomes occluded by B. Consequently, as a ray traverses through B, density accumulation occurs along that ray. When the ray subsequently intersects with A, it won't significantly affect the color because the density has already been accumulated. However, if the ray extends into empty space and encounters another object, it will have an impact on the final color because, in this case, density accumulation has not yet taken place, and the first object encountered will influence the color as the ray progresses. In other words, it quantifies the probability that the ray travels from ```tn``` to ```t``` without encountering any other particles along its path.

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/d408153b-0ec9-443f-996b-778f56b6aa6a" width="20%" />
</p>

1. To compute the color of a camera array that passes through the volume we need to estimate a continuous 1d line integral along that ray.

2. They do this by querying the MLP at multiple sample points within the range of starting and ending distances, denoted as t1 and tn.

3. The author does not solve this integration numerically but instead uses the ```quadrature rule```.

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/e74f3013-bae2-487d-8485-f8d73f7643a9" width="30%" />
</p>

4. This estimation process computes the color ![CodeCogsEqn (12)](https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/38bc9dd1-b888-4520-b263-e9f2f5158c64) of any camera ray by summing up contributions from each segment of the ray's path.

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/6eec67d0-3eb1-4b24-bbd5-73b575671d6e" width="25%" />
</p>

5. Each contribution includes the color of the segment ![CodeCogsEqn (13)](https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/8e1d0b01-a684-45d9-877c-50ddae651da2), which is weighted by the accumulated transmittance, ![CodeCogsEqn (14)](https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/2f47303e-24b2-47f4-a512-3f96851a521c), which computes how much light is blocked earlier along the ray _and_ ![CodeCogsEqn (15)](https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/8262a74f-dfc8-43e4-918b-585bdcf9d1a3) which is how much light is contributed by ray segment i, which is a function of the segment's length and its estimated volume density.

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/4547ce16-bb95-49cc-a316-b5d1865d368a" width="17%" />
</p>

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/52876cd2-5491-4278-b0be-9ade4c42d677" width="17%" />
</p>

The author also argues that they allow the color of any 3D point to vary as a function of the **viewing direction** as well as **3D position**. If we change the direction inputs for a fixed (x,y,z) location, we can visualize what view-dependent effects have been encoded by the network. Below is a visualization of two different points in a synthetic scene. It demonstrates how for a fixed 3D location adding view directions as an extra input allows the network to represent realistic view-dependent appearance effects


<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/2cbb0834-8d8e-4ef1-a7e1-0c0eff2d5092" width="60%" />
</p>






### 1.3 Improvement 1: Positional Encoding


### 1.4 Improvement 2: Hierarchical Volume Sampling

----------

## 2. 3D Model using Blender






---------------

## 3. Training NeRF



--------------------------






## References
1. https://www.youtube.com/watch?v=CRlN-cYFxTk&ab_channel=YannicKilcher
2. https://www.youtube.com/watch?v=JuH79E8rdKc&ab_channel=MatthewTancik
3. https://www.youtube.com/watch?v=LRAqeM8EjOo&ab_channel=BENMILDENHALL
4. https://www.fxguide.com/fxfeatured/the-art-of-nerfs-part1/?lid=7n16dhn58fxs
5. https://www.youtube.com/watch?v=CRlN-cYFxTk&t=1745s&ab_channel=YannicKilcher
6. https://www.fxguide.com/fxfeatured/the-art-of-nerfs-part-2-guassian-splats-rt-nerfs-stitching-and-adding-stable-diffusion-into-nerfs/
7. https://www.youtube.com/watch?v=nCpGStnayHk&ab_channel=TwoMinutePapers
8. https://www.youtube.com/watch?v=4NshnkzOdI0&list=PLlrATfBNZ98edc5GshdBtREv5asFW3yXl&index=2&ab_channel=TheCherno
9. https://www.youtube.com/watch?v=AjkiBRNVeV8&ab_channel=TwoMinutePapers
10. https://www.youtube.com/watch?v=NRmkr50mkEE&ab_channel=TwoMinutePapers
11. https://www.youtube.com/watch?v=ll4_79zKapU&ab_channel=BobLaramee
12. https://www.youtube.com/watch?v=g50RiDnfIfY&t=112s&ab_channel=PyTorch
13. https://www.peterstefek.me/nerf.html#:~:text=NeRF%20relies%20on%20a%20very,is%20useful%20for%20understanding%20NeRF.
14. https://datagen.tech/guides/synthetic-data/neural-radiance-field-nerf/#
15. https://dtransposed.github.io/blog/2022/08/06/NeRF/
