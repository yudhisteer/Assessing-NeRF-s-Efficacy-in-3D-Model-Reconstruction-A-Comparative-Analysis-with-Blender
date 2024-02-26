import os
import numpy as np
from numpy import arange, pi, sin, cos, arccos
import json
import bpy 


def write_intrinsics(N):
    # Write intrinsics to files
    train_cnt = 0
    test_cnt = 0
    for j in range(N):

        root = 'D://downloads//blender//image_extraction/' #change the path to your main_folder path

        prefix = '/train' if (j + 1) % 10 else '/test'
        cnt = train_cnt if (j + 1) % 10 else test_cnt
        train_cnt += (1 if (j + 1) % 10 else 0)
        test_cnt += (0 if (j + 1) % 10 else 1)

        pixel_size = 36 / 400  # mm
        f = 120 / pixel_size
        intrinsics = [f, 0.0, 200, 0.0,
                      0.0, f, 200, 0.0,
                      0.0, 0.0, 1.0, 0.0,
                      0.0, 0.0, 0.0, 1.0]

        with open(root + prefix + '/intrinsics/' + f'{prefix}_{cnt}.txt', 'w') as f:
            for i in range(len(intrinsics) - 1):
                f.write(str(intrinsics[i]))
                f.write('\n')
            f.write(str(intrinsics[-1]))


def write_pose(prefix, cnt, c2w: np.array):
    root = 'D://downloads//blender//image_extraction/' #change the path to your main_folder path

    with open(root + prefix + '/pose/' + f'{prefix}_{cnt}.txt', 'w') as f:
        pose = c2w.reshape(-1).tolist()
        for i in range(len(pose) - 1):
            f.write(str(pose[i]))
            f.write('\n')
        f.write(str(pose[-1]))


def rotate_and_render(output_dir, output_file_pattern_string='render%d.jpg',
                      rotation_steps=32, rotation_angle=360.0, radius=3, rename=True):
                          
    #make_files() #unused function
    subject = bpy.data.objects["Camera"] #pass the camera as subject for rotation
    write_intrinsics(rotation_steps)

    original_rotation = subject.rotation_euler.copy()
    original_location = subject.location.copy()

    train_cnt = 0
    test_cnt = 0

    n = rotation_steps
    n *= 2  # We will kill z < 0
    goldenRatio = (1 + 5 ** 0.5) / 2
    i = arange(0, n)
    theta = 2 * pi * i / goldenRatio
    phi = arccos(1 - 2 * (i + 0.5) / n)
    # phi = phi * 0 + np.pi / 2
    cond = (phi > (np.pi / 2)) & (phi < (3 * np.pi / 2))
    phi = phi[~cond]
    theta = theta[~cond]
    x, y, z = radius * cos(theta) * sin(phi), radius * sin(theta) * sin(phi), radius * cos(phi);

    bpy.context.scene.render.resolution_x = 400
    bpy.context.scene.render.resolution_y = 400

    for i, step in enumerate(range(0, rotation_steps)):

        subject.location = (x[i], y[i], z[i])
        subject.rotation_euler[0] = phi[i]  # np.pi / 2# + np.radians(theta[i])
        subject.rotation_euler[1] = 0.
        subject.rotation_euler[2] = np.pi / 2 + theta[i]
        # subject.rotation_euler[2] = np.radians(step * (rotation_angle / rotation_steps))
        print('position', subject.location)
        print('orientation', subject.rotation_euler)
        print()

        if rename:
            if (step + 1) % 10:  # Train
                bpy.context.scene.render.filepath = os.path.join(output_dir, f'train_{train_cnt}')
                write_pose('train', train_cnt, np.array(subject.matrix_world))
                train_cnt += 1
            else:  # Test
                bpy.context.scene.render.filepath = os.path.join(output_dir, f'test_{test_cnt}')
                write_pose('test', test_cnt, np.array(subject.matrix_world))
                test_cnt += 1
        else:
            bpy.context.scene.render.filepath = os.path.join(output_dir, (output_file_pattern_string % step))
            
        bpy.context.scene.render.engine = 'CYCLES' #render with cycles engine to improve quality
        bpy.ops.render.render(write_still=True)
        
    bpy.context.scene.render.film_transparent = True #render with transparent background
    subject.rotation_euler = original_rotation
    subject.location = original_location
    
    
rotate_and_render("D://downloads//blender//image_extraction//images") #path of the images folder 