# Assessing NeRF's Efficacy in 3D Model Reconstruction: A Comparative Analysis with Blender
[NeRF](https://arxiv.org/pdf/2003.08934.pdf) in 2020 changed 3D reconstruction by showing how to _effectively optimize neural radiance fields to render photorealistic novel views of scenes with complicated geometry and appearance_. In my previous project, I showed how to do 3D reconstruction with Multi-View Stereo (MVS) and Structure from Motion (SfM) which can generate "decent results" However, NeRF uses a fully connected (non-convolutional) deep network to represent a scene that achieved state-of-the-art results.

After taking the "Intro to Blender" by Studio X at UofR, I wanted to work more on my blender model using NeRF. I wanted to see how close we can match the 3D blender model if I randomly only generate images of the 3D model along with their intrinsic and extrinsic parameters, and input them into a NeRF model. This project focuses on building a vanilla-NeRF model from scratch based on the NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (2020) paper. Note that the latest NeRFs methods use hashing as in [instant-NGP](https://docs.nerf.studio/nerfology/methods/instant_ngp.html) since it significantly reduces the required number of layers in MLP. Also, for purely Visual reconstruction [Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) is significantly superior to NeRFs. However, this remains a good exercise to understand the mechanics behind the paper which changed 3D reconstruction.

## Dataset
One drawback of NeRF (the one which we will implement) is that once trained, we cannot use the model to 3D reconstruct another object. That is, suppose we have images of pineapple and we use NeRF to 3D reconstruct the pineapple. We cannot use that same model to 3D reconstruct a banana for example. This means, that if we want to 3D reconstruct the banana, we need to train it specifically for these images and then extract the mesh. Hence, we generate our images from Blender by running the [Blender.py](https://github.com/yudhisteer/Assessing-NeRF-s-Efficacy-in-3D-Model-Reconstruction-A-Comparative-Analysis-with-Blender/blob/main/Blender.py) script inside Blender which has 100 images (90 training, 10 testing) by rotating the camera on a sphere of radius 3. For each image, we also extract the intrinsic and extrinsic parameters associated. Below is the associated file format for the "[Clown](https://github.com/yudhisteer/Assessing-NeRF-s-Efficacy-in-3D-Model-Reconstruction-A-Comparative-Analysis-with-Blender/tree/main/images)" dataset:

```python
Clown/
│
├── images/
│   ├── train_0.png/
│   ├── train_1.png/
│   └── .../
│   ├── test_0.png/
│   ├── test_1.png/
│   └── .../
│
├── train/
│   ├── intrinsics/
│   │   ├── train_0.txt/
│   │   ├── train_1.txt/
│   │   ├── train_2.txt/
│   │   └── .../
│   │
│   └── pose/
│       ├── train_0.txt/
│       ├── train_1.txt/
│       ├── train_2.txt/
│       └── .../           
│
└── test/
     ├── intrinsics/
     │   ├── test_0.txt/
     │   ├── test_1.txt/
     │   ├── test_2.txt/
     │   └── .../
     │
     └── pose/
         ├── test_0.txt/
         ├── test_1.txt/
         ├── test_2.txt/
         └── .../
```

## Plan of Action

0. [Prerequisites](#p)
      - [Ray Tracing](#rt)
      - [Ray Casting](#rc)
      - [Ray Marching](#rm)
      - [NeRFing a sphere: Part I](#ns1)

1. [Understanding NeRF](#un)
     - [Volumetric Scene Representation](#vsr)
     - [Volume Rendering](#vr)
     - [NeRFing a sphere: Part II](#ns2)
     - [Improvement 1: Positional Encoding](#pe)
     - [Improvement 2: Hierarchical Volume Sampling](#hs)

2. [3D Model using Blender](#mub)


3. [Training NeRF](#tn)
      - [Fully Connected Network](#fcn)
      - [Novel View Synthesis](#nvs)


---------------
<a name="p"></a>
## 0. Prerequisites
Before we dive into NeRF, let's get a very brief introduction to the different techniques used in computer graphics to render 3D scenes by simulating the way light interacts with objects.

<a name="rt"></a>
### 0.1 Ray Tracing

Ray tracing works by calculating how light interacts with objects in a 3D scene. It's like simulating the path of a beam of light as it bounces off surfaces, refracts through materials, and creates shadows. This technique tracks rays of light from the viewer's eye through each pixel on the screen, considering complex interactions such as **reflections** and **refractions**. This results in **highly detailed** and **photorealistic images**.


<div style="text-align: center;">
  <video src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/deffeb2e-5ce7-4640-915c-ac198e887afd" controls="controls" style="max-width: 730px;">
  </video>
</div>
<div align="center">
    <p>Video source: <a href="https://www.youtube.com/watch?v=NRmkr50mkEE&t=530s&ab_channel=TwoMinutePapers">Ray Tracing: How NVIDIA Solved the Impossible!</a></p>
</div>

<a name="rc"></a>
### 0.2 Ray Casting
Ray casting is more straightforward. Imagine you're taking a photo with a camera. For each pixel on the screen, a single ray is sent out from your eye, and it checks if it **hits** anything in the scene. This technique is quick because it doesn't consider complex lighting effects like reflections or global illumination. It's suitable for **real-time applications** where speed is essential, such as early video games and simple simulations.

<div style="text-align: center;">
  <video src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/8bfce500-6c63-47ff-8fe3-ed082298fb02" controls="controls" style="max-width: 730px;">
  </video>
</div>
<div align="center">
    <p>Video source: <a href="https://www.youtube.com/watch?v=5xyeWBxmqzc&list=PLlYT7ZZOcBNA1hVBjkKFMnW0YDDODdy40&ab_channel=FinFET">How to make a simple 3D* game in Python from scratch - Ray casting</a></p>
</div>

<a name="rm"></a>
### 0.3 Ray Marching
Ray marching is like exploring a scene step by step. It sends out a ray and takes **small steps** along it, checking for objects or changes in the scene. This is useful for creating unusual and mathematical shapes or for rendering things like **clouds** or **fractals** where the structure is **complex** and not always easy to calculate all at once.




<div style="text-align: center;">
  <video src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/a2ee6682-de6d-4d76-a3b5-b72b108733d9" controls="controls" style="max-width: 730px;">
  </video>
</div>
<div align="center">
    <p>Video source: <a href="https://www.youtube.com/watch?v=t8Aoq5Rkyf0&ab_channel=Auctux">2D Ray Marching Visualization : Python</a></p>
</div>



Instead of directly calculating intersections and shading like traditional ray casting, NeRF uses a **neural network** to learn the 3D representation of the scene. The neural network takes the ```rays'``` **directions** and **origins** as **input** and **predicts** the ```3D scene's appearance (R,G,B)``` and ```structure (Density)``` at those points.


<a name="ns1"></a>
### 0.4 NeRFing a sphere - Part I
Before diving deep into NeRFing a ```3D``` model, I want to take the simplest example: **a sphere**. We will first try to apply the principles as shown in the NeRF paper to 3D model a sphere. In this section, we will apply **ray-casting** techniques in order to create a sphere then we will improve it in the following sections.

We start by defining the difference between a **line** and a **ray**. while a line is an infinite straight path extending in **both directions**, a ray has an **origin** and a **direction vector** that extends infinitely in **one** direction. The equation of a ray can be modeled as a **parametric equation**:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/56095af0-764d-431b-8361-3d46a55947de"/>
</p>

where ```o``` is the **origin**, ```d``` is the **direction vector**, ```t``` is a **parameter** that varies along the ray and determines different points along its path, and ```r``` is a **position vector** representing any point on the ray.

Let's look at an example of how we can apply this equation. Suppose we have the origin of a vector at ```(2,3)``` with a direction vector of ```(1,1)```. We want to know the position vector of that ray when ```t=5```. Using the equation above:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/f8ad3b8d-3755-4600-9268-25f025f2c244"/>
</p>

Note that ```(7,8)``` is the horizontal and vertical displacement in the ```x``` and ```y``` directions. That is we have moved ```7.07``` units along the ray, using Pythagoras theorem, from point ```A``` to point ```C``` and not ```5 units``` as specified by ```t```. If we want to move ```5 units``` along that ray, then we need to **normalize** our ```direction vector``` into a **unit vector**. 

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/6a0529b3-5c7b-4366-9b91-84d0358605af"/>
</p>

We then re-calculate our position vector which is now ```(5.5, 6.5)```. If we check again with **Pythagoras theorem**, then we have indeed moved ```5 units``` along that ray.

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


We will first work on the mathematical calculations of how we can model a circle. Since it is easier to work in ```2D```, when modeling for a ```3D``` sphere we will just need to add a ```z``` component. We will change our equation of a ray with different variables to avoid any notation confusion in the future. Note that I also separated it into their ```x-y``` components:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/47eff1bc-4cd9-4ded-aa9a-7fbd8ac387f6"/>
</p>

Below is the equation of a circle where a and b are the center and r is the radius. 

<p align="center">
  <img src=https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/8985882e-d998-4d8f-b8b6-35059854216b/>
</p>

Suppose we have a circle centered at the origin with radius ```3```. We also have a ray with an origin ```(-4,4)``` with a direction vector of ```(-1,-1)```, we want to know if that ray will **intersect** with the circle, and if so, **where**? 

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

We start by replacing the ```x``` and ```y``` components of our ray equation into the equation of the sphere:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/80b77287-c213-46ac-9797-6c09233de40c"/>
</p>

We now expand the equation and remove ```t``` outside the bracket:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/39e69910-3faa-4373-ae98-4b5ed7e9547e"/>
</p>

In order to solve this quadratic equation, we can use the **quadratic formula**:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/01d96934-b062-4797-a5ca-2f0abcdb70cc"/>
</p>

with the **discriminant** being: 

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/309a646d-c9ed-4c7b-97bc-4786a0e9fed6"/>
</p>

where:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/07562b4e-924f-4110-9b21-2c93b54052e0"/>
</p>

Note that we can first check if we have any solution at all by plugging in the values into the **discriminant**. Normally, if the ```discriminant = 0```, then we have **one solution** such that the line is **tangent** to the circle, if the ```discriminant > 0```, then we have **2 solutions** with the line intersecting the circle at **two distinct points** and finally, if the ```discriminant < 0```, then we have **0 solutions**, with the line **not intersecting** the circle at all. Below is a graphical representation of it:


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

We then solve for ```t``` and plug the values of the latter into our equation for the ray where we get the ```x``` and ```y``` values for the point of intersections which are: ```(-2,12, 2.12)``` and ```(2.12, -2.12)```. Now let's implement it in Python but for a **sphere**. Our quadratic formula will change to:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/dbb7704b-9e65-4655-9a54-2be339477ece"/>
</p>

We start by creating a **class** for the sphere whereby we will first compute the discriminant and then check if the latter ```>= 0```, then we will color the pixel ```red```. Note that previously, we already have created our rays which will originate from ```(0,0,0)``` and project downwards the ```z-axis```. Our goal will be similar to that of the circle above, find where the rays intersect and then color the pixel.

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/3df8756d-ac8e-498f-8cba-5654d19458e8" width="50%"/>
</p>

Note that previously, we set the origin as the circle at ```(0,0)```, here we will incorporate the center as a **variable** ```(cx, cy, cz)``` which can be changed.

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
Here's the result. When viewed on a ```2D``` plane, it appears to be a **circle** but with Plotly in ```3D``` we indeed confirm we have created a **sphere**. Also, notice that our sphere is **hollow**. That is, we have only points on the surface of the sphere and none on the inside. This makes sense as we are only taking into account the point of intersections of the ray and the sphere. In the next iteration, we will change this to have a **solid sphere** and assign density to the points inside.

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/8fc823ee-7deb-4d87-bcec-281331a4591a" width="45%"/>
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/f1d1d661-34eb-4155-a9f1-a76b332b7ebd" width="45%">
</p>

Let's explore the NeRF paper first before improving our sphere further.



----------------------------
<a name="un"></a>
## 1. Understanding NeRF

<a name="vsr"></a>
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
    <p>Image Source: <a href="https://arxiv.org/pdf/2003.08934.pdf">NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis</a></p>
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





<a name="vr"></a>
### 1.2 Volume Rendering
Recall that density, σ, can be binary, where it equals ```1``` if the point is on the object's surface, i.e., it intersects with the scene geometry, and ```0``` if it is in empty space. Hence, everywhere in space, there is a value that represents density and color at that point in space.

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/5e70e0bc-ac28-4816-b27f-f53b0d97b501" width="60%" />
</p>
<div align="center">
    <p>Image Source: <a href="https://arxiv.org/pdf/2003.08934.pdf">NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis</a></p>
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
<div align="center">
    <p>Image Source: <a href="https://www.youtube.com/watch?v=HfJpQCBTqZs&t=1576s">Understanding and Extending Neural Radiance Fields</a></p>
</div>

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

The author also argues that they allow the color of any 3D point to vary as a function of the **viewing direction** as well as **3D position**. If we change the direction inputs for a fixed (x,y,z) location, we can visualize what view-dependent effects have been encoded by the network. Below is a visualization of two different points in a synthetic scene. It demonstrates how for a fixed 3D location adding view directions as an extra input allows the network to represent realistic view-dependent appearance effects. P.S. Kudos to Maxime Vandegar for the explanation.


<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/2cbb0834-8d8e-4ef1-a7e1-0c0eff2d5092" width="60%" />
</p>
<div align="center">
    <p>Image Source: <a href="https://arxiv.org/pdf/2003.08934.pdf">NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis</a></p>
</div>



<a name="ns2"></a>
### 1.3 NeRFing a sphere: Part II
Signed distance functions, SDFs,  when passed the coordinates of a point in space, return the shortest distance between that point and some surface. The sign of the return value indicates whether the point is inside that surface or outside. For our case, points inside the sphere will have a ```distance from the origin < the radius```, points on the sphere will have ```distances = equal to the radius```, and points outside the sphere will have ```distances > than the radius```.

We will change our class sphere such that this time we won't compute for intersections of rays with the sphere, but instead sample points along the rays and check if the point is less than the radius. If so, we will assign a color to that point and a density value.

```python
class Sphere ():

    def __init__(self, center, radius, color):
        self.center = center
        self.radius = radius
        self.color = color

    def intersect(self, point):

        '''
        :param point: [batch_size, 3]. It is the sampled point along a ray
        :return: color, density
        '''

        # Center components
        cx = self.center[0]
        cy = self.center[1]
        cz = self.center[2]

        # separtate point into x-y-z components
        x_coor = point[:, 0]
        y_coor = point[:, 1]
        z_coor = point[:, 2]

        # any point less than radius^2 are in the sphere
        # x^2 + y^2 + z^2 < r^2
        condition = (x_coor-cx)**2 + (y_coor-cy)**2 + (z_coor-cz)**2 < self.radius**2

        # store colors and density for each ray.
        num_rays = point.shape[0] #16000
        colors = np.zeros((num_rays, 3))
        density = np.zeros((num_rays, 1))

        # Iterate over each ray and check the condition
        for i in range(num_rays):
            if condition[i]: # if condition[i] = true
                colors[i] = self.color
                density[i] = 10

        return colors, density
```

Next, the author talks about using a stratified sampling approach whereby they partition ```[tn, tf]``` into **N evenly-spaced bins** and we also need to calculate the distance between the adjacent samples which is equal to **delta**. We implement it as follows:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/2fe30b96-881b-4eec-9ea3-d832ffd6768b" />
</p>

```python
    # divide our ray at equally spaced intervals
    t = torch.linspace(tn, tf, bins).to(device)

    # calculate delta: t_i+1 - t_i ## distance between adjacent samples
    delta = torch.cat((t[1:] - t[:-1], torch.tensor([1e10], device=device))) #for the last delta value we set to infinity (1e10)
```
We define our equation of ray:

```python
    # Equation of rays
    ray = ray_origin + t * ray_direction
```

We calculate the RGB color and density at that sampled point:

```python
    # calculate colors, and density at the sampled point
    colors, density = model.intersect(ray_reshape)
```

We compute alpha as follows:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/907fab98-3779-4b2b-b012-ce9e27044851" />
</p>

```python
    # compute alpha
    alpha = 1 - torch.exp(- torch.from_numpy(density) * delta.unsqueeze(0))
```

Next, we compute the accumulated transmittance using the equation below:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/35579906-0d9d-4c96-ae8c-354c019b1071" />
</p>

```python
def accumulated_transmittance(alpha):
    # T = Π(1-alpha) {j=1, i-1}
    T = torch.cumprod((1-alpha), 1)
    # # shift everything to write as j starts at 1 (and not 0)
    T[:, 1:] = T[:, :-1]
    # # set first iteration = 1 (e^0 = 1)
    T[:, 0] = 1
    return T

# compute accumulated transmittance
T = accumulated_transmittance(alpha) #([160000, 100])
```
Finally, we compute the expected color using the equation below:

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/6eec67d0-3eb1-4b24-bbd5-73b575671d6e" width="25%" />
</p>

```python
    # computer expected color
    expected_color = (T.unsqueeze(-1) * alpha.unsqueeze(-1) * torch.from_numpy(colors)).sum(1) #sum along bins
```

From the image below, we observe that the first sphere was from the first iteration which shows no rendering. The second sphere is a rendered one which we just did. Notice how it gives a more 3D realistic rendering of the object. one important thing to note here is that our sphere is no longer hollow. That is, we can see we have points inside the sphere as shown in the last diagram. It seems we have layers of circles with reducing radii that form the sphere, as we go towards the top. This is due to the N evenly-spaced sampled points from our rays.

<table>
  <tr>
    <td><img width="446" alt="image" src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/9dbed1ff-9780-4945-92e5-50aa5d06f0d8"></td>
    <td><img width="437" alt="image" src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/308b2e8b-26d3-4541-a5c9-ff1975e5f234"></td>
    <td><img width="754" alt="image" src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/177e20a2-57f4-4773-bf23-acc600c66042"></td>

  </tr>
</table>


The author argues that the function to calculate the expected color is **differentiable**. Hence, we will test this by first creating a **ground truth** sphere of color ```red```.

```python
    # parameters for sphere
    center = np.array([0., 0., -2.])
    radius = 0.1
    color = np.array([1., 0., 0.]) #red

    # create model with parameters
    model = Sphere(center, radius, color)
    # create target image
    b = rendering(model, ray_origin, ray_direction, tn=1.0, tf=2.0, bins=100, device='cpu')
```

We will then initialize a second color that will need to be **optimized**. We will set it to ```green``` with the parameter ```requires_grad=True```.

```python
    # Optimization of color
    color_to_optimize = torch.tensor([0., 1., 0.], requires_grad=True) #green
```

We then set our optimizer to **Stochastic Gradient Descent (SGD)**, calculate the **loss** between the _ground truth_ and our _predicted value_, and do **backpropagation** to update the color. 

```python
    # create an SGD optimizer with the color_to_optimize tensor as the parameter to optimize
    optimizer = torch.optim.SGD([color_to_optimize], lr=1e-1)
    # list to store training losses
    training_loss = []

    for epoch in range(200):
        # create a sphere model with the parameters
        model = Sphere(center, radius, color=color_to_optimize)
        # render the scene - Ax
        Ax = rendering(model, ray_origin, ray_direction, tn=1.0, tf=2.0, bins=100, device='cpu')
        # calculate the loss as the mean squared difference between Ax and the target image b
        loss = ((Ax - b) ** 2).mean()
        # zero the gradients in the optimizer
        optimizer.zero_grad()
        # Compute gradients using backpropagation
        loss.backward()
        # update model parameters using the optimizer
        optimizer.step()
```
We train for ```200``` epochs and plot the resulting image after each ```10``` epochs. Below is the result

<img width="1194" alt="image" src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/c9cf0923-610f-4948-ac20-a39ad80d2855">

Notice how we started with a green sphere and after each ```10```iteration, we can see the changes from **green** to **red**. By iteration ```120``` we have a fully red sphere.

<a name="pe"></a>
### 1.3 Improvement 1: Positional Encoding
The author argues that when the neural network operates directly on the input coordinates ```(x,y,z)``` and the viewing direction,**d**, the resulting renderings struggle to capture fine details in color and geometry. That is, the neural network is **not good** at accurately capturing and displaying these small, detailed changes in color and shape, which can be important for realistic and detailed 3D scene rendering. 

What they did instead was to use ```high-frequency functions``` to map the inputs to a ```higher-dimensional space``` before passing them to the MLP and this accurately captured and modeled the data with intricate, **high-frequency variations**. As shown in the image below, they map the _spatial position_ and _viewing direction_ (after converting from a **spherical** to a **cartesian** coordinates system) into a higher dimensional space using the ```sine``` and ```cosine``` functions. 

<p align="center">
  <img src="https://github.com/yudhisteer/Training-a-Neural-Radiance-Fields-NeRF-/assets/59663734/7bb993ad-4cff-4908-80bd-863f3ce3710f" width="70%" />
</p>

Note that, for the _3D spatial location_ they use **L (frequency of encoding) = 10**, which means they will use frequency functions up to ```(sin(2^9(x)), cos(2^9(x)))```, and similarly for the _viewing direction_ they use **L = 4**, which ends at ```((sin(2^3(x)), cos(2^3(x)))```. One important thing to observe is that when mapping  into high dimensional space, we are **only** predicting the ```color (r,g,b)``` values alone and **not** the ```density```.

Let's see how we can code this:

```python
def positional_encoding(x: torch.Tensor, L: int) -> torch.Tensor:
    # to store encodings
    encoding_components = []

    # loop over encoding frequencies up to L
    for j in range(L):
        # Calculate sine and cosine components for each frequency
        encoding_components.append(torch.sin(2 ** j * x))
        encoding_components.append(torch.cos(2 ** j * x))

    # concatenate original input with encoding components
    encoded_coordinates = torch.cat([x] + encoding_components, dim=1)

    return encoded_coordinates
```

Note that the output shape is ```(N, 3 + 6 * L)``` as the original input **x** has 3 features ```(x, y, z)``` and for each of the 3 spatial dimensions (x, y, z), we apply ```L``` frequencies of encoding, resulting in ```2 x L```. Since we have 3 spatial dimensions, we have a total of ```3 x (2 x L) = 6 x L``` encoding components in total. In the end, we concatenate our original features with  encoding components, hence ```3 + 6 x L```.


<a name="hs"></a>
### 1.4 Improvement 2: Hierarchical Volume Sampling

A second improvement that the authors suggest to efficiently render 3D scenes is hierarchical volume sampling. It involves creating two levels of volume sampling: a **coarse** and a **fine** one. Initially, the scene is sampled at a coarse level to understand the basic **structure** and **distribution** of light and matter. The algorithm then performs a more **detailed** sampling at a fine level in areas of interest. 

This method is like being smart about painting a picture. Instead of painting every part in detail from the start, you first figure out which parts of the picture are most important. Then, you spend more time and effort on those important parts to make them look really good. This way, you finish the picture faster and it looks better because you focus on the parts that matter the most.

I will omit this part in the code for a simpler version of NeRF. This part is still a work in improvement for me. Patience!

----------
<a name="mub"></a>
## 2. 3D Model using Blender

In the fall of 2023, I took an AR/VR class where I learned Blender from [Studio X](https://www.library.rochester.edu/spaces/studio-x) at the University of Rochester. Below is my 3D model from that class which is a combination of 3D meshes like cylinder, cube, cone, torus, and ico sphere. 

<div style="text-align: center;">
  <video src="https://github.com/yudhisteer/Assessing-NeRF-s-Efficacy-in-3D-Model-Reconstruction-A-Comparative-Analysis-with-Blender/assets/59663734/19e96775-bb2e-4e3a-9a0f-3e23f404a32a" controls="controls" style="max-width: 730px;">
  </video>
</div>

I will then run a script inside Blender that will take pictures at different angles of the 3D model and register the intrinsic and extrinsic parameters of the camera associated with each image.

![v2](https://github.com/yudhisteer/Assessing-NeRF-s-Efficacy-in-3D-Model-Reconstruction-A-Comparative-Analysis-with-Blender/assets/59663734/b6b1f5f9-77a0-4850-975c-5b2f06b78b24)

---------------
<a name="tn"></a>
## 3. Training NeRF

<a name="fcn"></a>
### 3.1 Fully Connected Network
Up to this point, we've discussed the high-level representation of the continuous 5D input through an MLP. Now, let's delve into the fully connected network and dissect its components. 

The architecture is quite simple: we have 8 fully connected ReLU layers, each with 256 channels. At the 5th layer, there's a skip connection. It's important to note that we take input from the **positional encoding of the input location**. In the 9th layer, we merge the 256-dimensional feature vector with the **positional encoding of the viewing direction**. The final output consists of 3 color channels and 1 density channel.

<p align="center">
  <img src="https://github.com/yudhisteer/Neural-Radiance-Fields-NeRF-on-custom-synthetic-datasets/assets/59663734/6d75a70b-d978-42b2-896e-7c36c72e3447" width="90%" />
</p>

#### 3.1.1 MLP Architecture
Let's set up the architecture and initialize the parameters. We are using the formula ```(N, 3 + 6 * L)``` to calculate the size of the ```in_features``` parameter. Though the paper states that the input is ```60```, the real input size is ```63```.

```python
    def __init__(self, L_pos=10, L_dir=4, hidden_dim=256):
        super(Nerf, self).__init()
        
        # Frequency of encoding
        self.L_pos = L_pos
        self.L_dir = L_dir

        # Fully connected layers
        # Block 1:
        self.fc1 = nn.Linear(L_pos * 6 + 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        # Block 2:
        self.fc6 = nn.Linear(hidden_dim + L_pos * 6 + 3, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, hidden_dim)
        self.fc9 = nn.Linear(hidden_dim, hidden_dim + 1)
        # Block 3:
        self.fc10 = nn.Linear(hidden_dim + L_dir * 6 + 3, hidden_dim // 2)
        self.fc11 = nn.Linear(hidden_dim // 2, 3)

        # Non-linearities
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
```

Now let's define the forward pass of the network step by step. First, we create encoded features for our spatial position **(x,y,z)** and viewing direction, **d** using the ```positional_encoding function``` we described before.

```python
        x_emb = self.positional_encoding(xyz, self.Lpos)  # [batch_size, Lpos * 6 + 3]
        d_emb = self.positional_encoding(d, self.Ldir)  # [batch_size, Ldir * 6 + 3]
```
#### 3.1.2 Block 1
The input location undergoes positional encoding and passes through a sequence of ```8``` fully connected ReLU layers, each comprising ```256``` channels.

```python
        ### ------------ Block 1:
        x = self.fc1(x_emb)  # [batch_size, hidden_dim]
        x = self.relu(x)
        print("Shape after fc1:", x.shape)

        x = self.fc2(x)
        x = self.relu(x)
        print("Shape after fc2:", x.shape)

        x = self.fc3(x)
        x = self.relu(x)
        print("Shape after fc3:", x.shape)

        x = self.fc4(x)
        x = self.relu(x)
        print("Shape after fc4:", x.shape)

        x = self.fc5(x)
        x = self.relu(x)
        print("Shape after fc5:", x.shape)
```
#### 3.1.2 Block 2
The author adopts the architectural approach from ```DeepSDF```, incorporating a **skip connection** that appends the input to the activation of the fifth layer. 

```python
        ### ------------ Block 2:
        x = self.fc6(torch.cat((x, x_emb), dim=1)) #skip connection
        x = self.relu(x)
        print("Shape after fc6:", x.shape)

        x = self.fc7(x)
        x = self.relu(x)
        print("Shape after fc7:", x.shape)

        x = self.fc8(x)
        x = self.relu(x)
        print("Shape after fc7:", x.shape)

        x = self.fc9(x)
        print("Shape after fc9:", x.shape)
```
#### 3.1.2 Block 3
For Block 3, an additional layer generates the **volume density**, which is rectified using a **ReLU** to ensure **non-negativity** and a ```256```-dimensional feature vector. This feature vector is **concatenated** with the positional encoding of the input viewing direction **d**, and the combined data is processed by an extra fully connected ReLU layer with ```128``` channels. Finally, a last layer, employing a **sigmoid activation**, produces the emitted ```RGB radiance``` at position **x**, as observed from a ray with direction **d**.

```python
        ### ------------ Block 3:

        # Extract sigma from x (last value)
        sigma = x[:, -1]

        # Density
        density = torch.relu(sigma)
        print("Shape of density:", density.shape)

        # Take all values from except sigma (everything except last one)
        x = x[:, :-1]  # [batch_size, hidden_dim]
        print("Shape after x:", x.shape)

        x = self.fc10(torch.cat((x, d_emb), dim=1))
        x = self.relu(x)
        print("Shape after fc10:", x.shape)

        color = self.fc11(x)
        color = self.sigmoid(color)
        print("Shape after fc11:", color.shape)
```
Let's check our code with **simulated data**:

```python
    # Simulated data
    xyz = torch.randn(batch_size=16, 3)
    d = torch.randn(batch_size=16, 3)
```
Below is the output:

```python
Shape of x_emb:  torch.Size([16, 63])
Shape of d_emb:  torch.Size([16, 27])
Shape after fc1: torch.Size([16, 256])
Shape after fc2: torch.Size([16, 256])
Shape after fc3: torch.Size([16, 256])
Shape after fc4: torch.Size([16, 256])
Shape after fc5: torch.Size([16, 256])
Shape after fc6: torch.Size([16, 256])
Shape after fc7: torch.Size([16, 256])
Shape after fc8: torch.Size([16, 256])
Shape after fc9: torch.Size([16, 257])
Shape after fc10: torch.Size([16, 128])
Shape after fc11: torch.Size([16, 3])
Density shape: torch.Size([16])
Color shape: torch.Size([16, 3])
```

We trained the model for ```5``` epochs, a batch size of ```1024```, and a learning rate of ```1e-3```  and below we see how the Mean Squared Error (MSE) loss curve starts at a relatively high-value then decreases sharply within the first few hundred iterations. The final loss value is quite low, close to zero, indicating good performance of the model on the training dataset

<p align="center">
  <img src="https://github.com/yudhisteer/Assessing-NeRF-s-Efficacy-in-3D-Model-Reconstruction-A-Comparative-Analysis-with-Blender/assets/59663734/a24bf2be-549a-4639-aa01-d717b9a45cc4" width="60%" />
</p>


--------------------------


<a name="nvs"></a>
### 3.2 Novel View Synthesis
To have a good 3D reconstruction, one parameter that we need to control is ```tn``` and ```tf``` which are the near and far bounds, respectively, along a ray. We generated our data from Blender by rotating the camera on a sphere with a radius of ```3```.  This means that the object can not be in the bounds outside ```tn = 0```, and ```tn = 6```. Also, the object is smaller than ```(2, 2)```, so an even better bound would be something like ```tn = 2``` and ```tf = 4```. As a rule, **tn = (radius - object_size - 0.2)** and **tf = (radius + object_size + 0.2)**.

We test our model on the test dataset which consists of 10 images and below are the results for different bounds. We also calculate the metric PSNR (Peak Signal-to-Noise Ratio) which is used to measure the quality of reconstruction of an image. It is expressed in decibels (dB) and higher values indicate better quality. A value of ```29.3``` dB was obtained.

<p align="center">
  <img src="https://github.com/yudhisteer/Assessing-NeRF-s-Efficacy-in-3D-Model-Reconstruction-A-Comparative-Analysis-with-Blender/assets/59663734/4b762118-f8b7-482b-98a4-842c027b70d7" />
</p>


<div style="display: flex; justify-content: center;">
      <table>
        <tr>
          <th>[tn, tf] = [8, 12]</th>
          <th>[tn, tf] = [1, 6]</th>
          <th>[tn, tf] = [2, 4]</th>
        </tr>
        <tr>
          <td><img src="https://github.com/yudhisteer/Assessing-NeRF-s-Efficacy-in-3D-Model-Reconstruction-A-Comparative-Analysis-with-Blender/assets/59663734/6b9a16db-e348-471b-b057-6608dd4e3c82" alt="First Image" height="300"/></td>
          <td><img src="https://github.com/yudhisteer/Assessing-NeRF-s-Efficacy-in-3D-Model-Reconstruction-A-Comparative-Analysis-with-Blender/assets/59663734/89adb6b7-fb98-40af-ad20-171bc178cc27" alt="Second Image" height="300"/></td>
          <td><img src="https://github.com/yudhisteer/Assessing-NeRF-s-Efficacy-in-3D-Model-Reconstruction-A-Comparative-Analysis-with-Blender/assets/59663734/d4d60c52-1435-46d1-a23e-cbb966ffdd89" alt="Third Image" height="300"/></td>
        </tr>
      </table>
</div>

Below are the extracted mesh for different bounds. As explained above, a bound between 2 and 4 would be more appropriate for this dataset.

<div style="display: flex; justify-content: center;">
    <table>
      <tr>
        <th>[tn, tf] = [8, 12]</th>
        <th>[tn, tf] = [6, 10]</th>
        <th>[tn, tf] = [3, 8]</th>
      </tr>
      <tr>
        <td><img src="https://github.com/yudhisteer/Assessing-NeRF-s-Efficacy-in-3D-Model-Reconstruction-A-Comparative-Analysis-with-Blender/assets/59663734/7d383d5a-f6f1-49bc-80dd-c2dd8af36698" alt="Image 1" height="320"/></td>
        <td><img src="https://github.com/yudhisteer/Assessing-NeRF-s-Efficacy-in-3D-Model-Reconstruction-A-Comparative-Analysis-with-Blender/assets/59663734/aa2c57d8-ff78-4c77-9068-3e4c995eec36" alt="Image 2" height="320"/></td>
        <td><img src="https://github.com/yudhisteer/Assessing-NeRF-s-Efficacy-in-3D-Model-Reconstruction-A-Comparative-Analysis-with-Blender/assets/59663734/1c1cb8e3-f0c7-4320-a25f-c1eaf9d2f7aa" alt="Image 3" height="320"/></td>
      </tr>
    </table>
</div>

Here's the result after ```5``` epochs. We clearly see the structure of the 3D model - the rectangular body, the two cylindrical eyes, the torus smile, the cone hat, and, the eco-sphere bottom - though it is not as refined as the blender model. This may be due to not implementing the hierarchical volume sampling feature of NeRF.  

<div style="text-align: center;">
  <video src="https://github.com/yudhisteer/Assessing-NeRF-s-Efficacy-in-3D-Model-Reconstruction-A-Comparative-Analysis-with-Blender/assets/59663734/80a30866-00ee-437a-86f8-1fb974db4914" controls="controls" style="max-width: 730px;">
  </video>
</div>


--------------------------
## Conclusion
In this project, we showed how to create a vanilla NeRF from scratch (though we emitted the Hierarchical Volume Sampling technique). We did mesh extraction but did not see color extraction for the 3D reconstructed model. The field of NeRF has since exploded and now has more advanced techniques such as Instant-NGP, iNeRF, KiloNeRF, FastNeRF, SqueezeNeRF, and so on. We needed known camera parameters (intrinsic and extrinsic) for this NeRF but now we also have NERF without known camera parameters. All in all, this remains a good experience to see really how NeRF works.


## References
[1] YouTube. (n.d.). Yannic Kilcher - NeRF: Representing Scenes as Neural Radiance Fields. [Video]. [https://www.youtube.com/watch?v=CRlN-cYFxTk&ab_channel=YannicKilcher](https://www.youtube.com/watch?v=CRlN-cYFxTk&ab_channel=YannicKilcher)

[2] YouTube. (n.d.). Matthew Tancik - NeRF: Neural Radiance Fields. [Video]. [https://www.youtube.com/watch?v=JuH79E8rdKc&ab_channel=MatthewTancik](https://www.youtube.com/watch?v=JuH79E8rdKc&ab_channel=MatthewTancik)

[3] YouTube. (n.d.). BEN MILDENHALL - [ECCV 2020] NeRF: Neural Radiance Fields (10 min talk). [Video]. [https://www.youtube.com/watch?v=LRAqeM8EjOo&ab_channel=BENMILDENHALL](https://www.youtube.com/watch?v=LRAqeM8EjOo&ab_channel=BENMILDENHALL)

[4] FXGuide. (n.d.). The Art of NeRFs Part 1. [https://www.fxguide.com/fxfeatured/the-art-of-nerfs-part1/?lid=7n16dhn58fxs](https://www.fxguide.com/fxfeatured/the-art-of-nerfs-part1/?lid=7n16dhn58fxs)

[5] YouTube. (n.d.). Yannic Kilcher - NeRF: Representing Scenes as Neural Radiance Fields (Extended). [Video]. [https://www.youtube.com/watch?v=CRlN-cYFxTk&t=1745s&ab_channel=YannicKilcher](https://www.youtube.com/watch?v=CRlN-cYFxTk&t=1745s&ab_channel=YannicKilcher)

[6] FXGuide. (n.d.). The Art of NeRFs Part 2: Gaussian Splats, RT NeRFs, Stitching, and Adding Stable Diffusion into NeRFs. [https://www.fxguide.com/fxfeatured/the-art-of-nerfs-part-2-guassian-splats-rt-nerfs-stitching-and-adding-stable-diffusion-into-nerfs/](https://www.fxguide.com/fxfeatured/the-art-of-nerfs-part-2-guassian-splats-rt-nerfs-stitching-and-adding-stable-diffusion-into-nerfs/)

[7] YouTube. (n.d.). Two Minute Papers - This Neural Network Learned To Look Around In Real Scenes! (NERF). [Video]. [https://www.youtube.com/watch?v=nCpGStnayHk&ab_channel=TwoMinutePapers](https://www.youtube.com/watch?v=nCpGStnayHk&ab_channel=TwoMinutePapers)

[8] YouTube. (n.d.). The Cherno - Rays and Spheres: The MATH! // Ray Tracing series. [Video]. [https://www.youtube.com/watch?v=4NshnkzOdI0&list=PLlrATfBNZ98edc5GshdBtREv5asFW3yXl&index=2&ab_channel=TheCherno](https://www.youtube.com/watch?v=4NshnkzOdI0&list=PLlrATfBNZ98edc5GshdBtREv5asFW3yXl&index=2&ab_channel=TheCherno)

[9] YouTube. (n.d.). Two Minute Papers - NVIDIA Did It: Ray Tracing 10,000 Times Faster!. [Video]. [https://www.youtube.com/watch?v=AjkiBRNVeV8&ab_channel=TwoMinutePapers](https://www.youtube.com/watch?v=AjkiBRNVeV8&ab_channel=TwoMinutePapers)

[10] YouTube. (n.d.). Two Minute Papers - Ray Tracing: How NVIDIA Solved the Impossible!. [Video]. [https://www.youtube.com/watch?v=NRmkr50mkEE&ab_channel=TwoMinutePapers](https://www.youtube.com/watch?v=NRmkr50mkEE&ab_channel=TwoMinutePapers)

[11] YouTube. (n.d.). Bob Laramee - Ray Casting versus Ray Tracing (Volumetric): A Quick and Convenient Comparison. [Video]. [https://www.youtube.com/watch?v=ll4_79zKapU&ab_channel=BobLaramee](https://www.youtube.com/watch?v=ll4_79zKapU&ab_channel=BobLaramee]

[12] YouTube. (n.d.). PyTorch - Rendering Volumes and Implicit Shapes in PyTorch3D. [Video]. [https://www.youtube.com/watch?v=g50RiDnfIfY&t=112s&ab_channel=PyTorch](https://www.youtube.com/watch?v=g50RiDnfIfY&t=112s&ab_channel=PyTorch)

[13] Peter Stefek. (n.d.). NeRF Explained. [Webpage]. [https://www.peterstefek.me/nerf.html](https://www.peterstefek.me/nerf.html#:~:text=NeRF%20relies%20on%20a%20very,is%20useful%20for%20understanding%20NeRF.)

[14] DataGen. (n.d.). Neural Radiance Field (NeRF). [Webpage]. [https://datagen.tech/guides/synthetic-data/neural-radiance-field-nerf/#](https://datagen.tech/guides/synthetic-data/neural-radiance-field-nerf/#)

[15] d'Transposed. (2022, August 6). NeRF. [Blog post]. [https://dtransposed.github.io/blog/2022/08/06/NeRF/](https://dtransposed.github.io/blog/2022/08/06/NeRF/)

[16] YouTube. (n.d.). Auctux - 2D Ray Marching Visualization: Python. [Video]. [https://www.youtube.com/watch?v=t8Aoq5Rkyf0&ab_channel=Auctux](https://www.youtube.com/watch?v=t8Aoq5Rkyf0&ab_channel=Auctux)

[17] Jamie Wong. (2016, July 15). Ray Marching and Signed Distance Functions. [Blog post]. [https://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/](https://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/)

[18] YouTube. (n.d.). CVPR 2022 Tutorial on Neural Fields in Computer Vision. [Video]. [https://www.youtube.com/watch?v=PeRRp1cFuH4](https://www.youtube.com/watch?v=PeRRp1cFuH4)

[19] NeRF Studio. (n.d.). Methods: Instant NGP. [Webpage]. [https://docs.nerf.studio/nerfology/methods/instant_ngp.html](https://docs.nerf.studio/nerfology/methods/instant_ngp.html)



