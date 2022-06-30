from xml.etree.ElementTree import ProcessingInstruction
import numpy as np
import camviz as cv


# ************************* Read depth maps **************************

# Flags
with_depth_input = True
crop_top = True
crop_per = 0.45

# Load predicted depth maps
data = np.load('demos/data/rgbd_sparse_hr.npz', allow_pickle=True)

# Parse dictionary information
intrinsics = data['intrinsics']
rgb = [np.copy(item) for item in data['rgb']]
rgb_copy = [np.copy(item) for item in data['rgb']]
depth = [np.copy(item) for item in data['depth']]
viz = [np.copy(item) for item in data['viz']]
depth_input = [np.copy(item) for item in data['depth_input']]

n = len(rgb) # Get number of frames
wh = rgb[0].shape[:2][::-1] # Get image resolution


# ********************** Create display windows **********************
draw = cv.Draw(wh=(1500, 800), width=1500, title='CamViz Pointcloud')
draw.add2Dimage('rgb', luwh=(0.00, 0.70, 0.50, 1.00), res=wh)
draw.add2Dimage('viz', luwh=(0.50, 0.70, 1.00, 1.00), res=wh)
draw.add3Dworld('wld_pred', luwh=(0.00, 0.00, 0.50, 0.70),
    pose=(7.25323, -3.80291, -5.89996, 0.98435, 0.07935, 0.15674, 0.01431))
draw.add3Dworld('wld_pred2', luwh=(0.50, 0.00, 1.00, 0.70),
    pose=(7.25323, -3.80291, -5.89996, 0.98435, 0.07935, 0.15674, 0.01431))

# Create camera from intrinsics and image dimensions (width and height)
camera = cv.objects.Camera(K=intrinsics, wh=wh)

def create_texture_for_one_frame(camera, rgb, rgb_copy, depth, viz, depth_input):
    # ********************* Project depth map to 3D *********************
    # Project depth maps from image (i) to camera (c) coordinates
    points_pred = camera.i2c(depth)
    # Project LiDAR input to 3d scene
    if with_depth_input:
        points_input = camera.i2c(depth_input)
        points_input[points_input==0] = points_pred[points_input==0]


    # ************************ Create color maps ************************
    # Create pointcloud colors
    rgb_clr = rgb.reshape(-1, 3)  # RGB colors
    viz_clr = viz.reshape(-1, 3)  # Depth visualization colors
    hgt_clr = cv.utils.cmaps.jet(-points_pred[:, 1])  # Height colors
    # Hybrid colors (to highlight LiDAR input)
    if with_depth_input:
        inp_clr = depth_input.reshape(-1, 3)
        hyb_rgb_clr = np.copy(inp_clr)
        hyb_rgb_clr[hyb_rgb_clr==0] = rgb_clr[hyb_rgb_clr==0]
        hyb_viz_clr = np.copy(inp_clr)
        hyb_viz_clr[hyb_viz_clr==0] = viz_clr[hyb_viz_clr==0]
        hyb_hgt_clr = np.copy(inp_clr)
        hyb_hgt_clr[hyb_hgt_clr==0] = hgt_clr[hyb_hgt_clr==0]


    # ********************* Create names for data *********************
    # Create names for data
    points_pred_name = 'pts_pred'
    rgb_name = 'rgb'
    viz_name = 'viz'
    rgb_clr_name = 'clr'
    viz_clr_name = 'viz'
    hgt_clr_name = 'hgt'
    if with_depth_input:
        hyb_rgb_clr_name = 'hyb_rgb'
        hyb_viz_clr_name = 'hyb_viz'
        hyb_hgt_clr_name = 'hyb_hgt'


    # ********************* Create display textures *********************
    # Create RGB and visualization textures
    draw.addTexture(rgb_name, rgb)  # Create texture buffer to store rgb image
    draw.addTexture(viz_name, viz)  # Create texture buffer to store visualization image
    # Create buffers to store data for display
    if with_depth_input:
        draw.addBufferf(points_pred_name, points_input) # Create data buffer to store hybrid depth points
        draw.addBufferf(hyb_rgb_clr_name, hyb_rgb_clr)  # Create data buffer to store hybrid rgb points color
        draw.addBufferf(hyb_viz_clr_name, hyb_viz_clr)  # Create data buffer to store hybrid viz points color
        draw.addBufferf(hyb_hgt_clr_name, hyb_hgt_clr)  # Create data buffer to store hybrid pointcloud heights
        color_dict = {0: hyb_rgb_clr_name, 1: hyb_viz_clr_name, 2: hyb_hgt_clr_name}
    else:
        draw.addBufferf(points_pred_name, points_pred)  # Create data buffer to store depth points
        draw.addBufferf(rgb_clr_name, rgb_clr)          # Create data buffer to store rgb points color
        draw.addBufferf(viz_clr_name, viz_clr)          # Create data buffer to store viz points color
        draw.addBufferf(hgt_clr_name, hgt_clr)          # Create data buffer to store pointcloud heights
        color_dict = {0: rgb_clr_name, 1: viz_clr_name, 2: hgt_clr_name}

    # Second view
    h_crop = int(crop_per * wh[1])
    points_pred_copy = np.copy(points_pred)
    # Crop to the top areas
    if crop_top:
        points_pred_copy = points_pred_copy.reshape(wh[1], wh[0], 3)
        points_pred_copy = points_pred_copy[h_crop:]
        points_pred_copy = points_pred_copy.reshape(-1, 3)
        rgb_copy = rgb_copy[h_crop:]
    rgb_copy_clr = rgb_copy.reshape(-1, 3)
    rgb_copy_name = 'rgb_copy'
    rgb_copy_clr_name = 'clr_copy'
    points_pred_copy_name = 'pts_pred_copy'
    draw.addTexture(rgb_copy_name, rgb_copy)
    draw.addBufferf(points_pred_copy_name, points_pred_copy)
    draw.addBufferf(rgb_copy_clr_name, rgb_copy_clr)

    return rgb_name, viz_name, points_pred_name, points_pred_copy_name, rgb_copy_clr_name, rgb_copy_name, color_dict


# ********************* Display data in loop *********************
color_mode = 0
count = 0
while draw.input():
    rgb_name, viz_name, points_pred_name, points_pred_copy_name, rgb_copy_clr_name, rgb_copy_name, color_dict \
         = create_texture_for_one_frame(camera, rgb[count], rgb_copy[count], depth[count], viz[count], depth_input[count])
    if draw.RETURN:
        color_mode = (color_mode + 1) % len(color_dict)
    draw.clear()
    draw['rgb'].image(rgb_name)
    draw['viz'].image(viz_name)
    draw['wld_pred'].size(2).points(points_pred_name, color_dict[color_mode])
    draw['wld_pred'].object(camera, tex=rgb_name)
    draw['wld_pred2'].size(2).points(points_pred_copy_name, rgb_copy_clr_name)
    draw['wld_pred2'].object(camera, tex=rgb_copy_name)

    draw.update(1)
    count = count + 1 if count < (n-1) else 0
