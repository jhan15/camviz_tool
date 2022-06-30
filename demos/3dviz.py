import numpy as np
import camviz as cv


# ************************* Read depth maps **************************

# Flags
with_depth_input = True
crop_top = True
crop_per = 0.45

# Show a few frames for test
test_show = True
frames_to_show = 20

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

if test_show and frames_to_show <= n:
    rgb = rgb[:frames_to_show]
    rgb_copy = rgb_copy[:frames_to_show]
    depth = depth[:frames_to_show]
    viz = viz[:frames_to_show]
    depth_input = depth_input[:frames_to_show]
    n = frames_to_show


# ********************** Create display windows **********************
draw = cv.Draw(wh=(1500, 800), width=1500, title='CamViz Pointcloud')
draw.add2Dimage('rgb', luwh=(0.00, 0.70, 0.50, 1.00), res=wh)
draw.add2Dimage('viz', luwh=(0.50, 0.70, 1.00, 1.00), res=wh)
draw.add3Dworld('wld_pred', luwh=(0.00, 0.00, 0.50, 0.70),
    pose=(7.25323, -3.80291, -5.89996, 0.98435, 0.07935, 0.15674, 0.01431))
draw.add3Dworld('wld_pred2', luwh=(0.50, 0.00, 1.00, 0.70),
    pose=(7.25323, -3.80291, -5.89996, 0.98435, 0.07935, 0.15674, 0.01431))


# ********************* Project depth map to 3D *********************
# Create camera from intrinsics and image dimensions (width and height)
camera = cv.objects.Camera(K=intrinsics, wh=wh)
# Project depth maps from image (i) to camera (c) coordinates
points_pred = [camera.i2c(d) for d in depth]
# Project LiDAR input to 3d scene
if with_depth_input:
    points_input = [camera.i2c(d) for d in depth_input]
    for input, pred in zip(points_input, points_pred):
        input[input==0] = pred[input==0]


# ************************ Create color maps ************************
# Create pointcloud colors
rgb_clr = [item.reshape(-1, 3) for item in rgb] # RGB colors
viz_clr = [item.reshape(-1, 3) for item in viz] # Depth visualization colors
hgt_clr = [cv.utils.cmaps.jet(-item[:, 1]) for item in points_pred] # Height colors
# Hybrid colors (to highlight LiDAR input)
if with_depth_input:
    inp_clr = [item.reshape(-1, 3) for item in depth_input]
    hyb_rgb_clr, hyb_viz_clr, hyb_hgt_clr = [], [], []
    inp_clr_copy = [np.copy(item) for item in inp_clr]
    for rgb_c, inp_c in zip(rgb_clr, inp_clr_copy):
        inp_c[inp_c==0] = rgb_c[inp_c==0]
        hyb_rgb_clr.append(inp_c)
    inp_clr_copy = [np.copy(item) for item in inp_clr]
    for viz_c, inp_c in zip(viz_clr, inp_clr_copy):
        inp_c[inp_c==0] = viz_c[inp_c==0]
        hyb_viz_clr.append(inp_c)
    inp_clr_copy = [np.copy(item) for item in inp_clr]
    for hgt_c, inp_c in zip(hgt_clr, inp_clr_copy):
        inp_c[inp_c==0] = hgt_c[inp_c==0]
        hyb_hgt_clr.append(inp_c)


# ********************* Create names for data *********************
# Create names for data
points_pred_name = ['pts_pred'+str(i) for i in range(n)]
rgb_name = ['rgb'+str(i) for i in range(n)]
viz_name = ['viz'+str(i) for i in range(n)]
rgb_clr_name = ['clr'+str(i) for i in range(n)]
viz_clr_name = ['viz'+str(i) for i in range(n)]
hgt_clr_name = ['hgt'+str(i) for i in range(n)]
if with_depth_input:
    hyb_rgb_clr_name = ['hyb_rgb'+str(i) for i in range(n)]
    hyb_viz_clr_name = ['hyb_viz'+str(i) for i in range(n)]
    hyb_hgt_clr_name = ['hyb_hgt'+str(i) for i in range(n)]


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
points_pred_copy = [np.copy(d) for d in points_pred]
# Crop to the top areas
if crop_top:
    points_pred_copy = [item.reshape(wh[1], wh[0], 3) for item in points_pred_copy]
    points_pred_copy = [item[h_crop:] for item in points_pred_copy]
    points_pred_copy = [item.reshape(-1, 3) for item in points_pred_copy]
    rgb_copy = [item[h_crop:] for item in rgb_copy]
rgb_copy_clr = [item.reshape(-1, 3) for item in rgb_copy]
rgb_copy_name = ['rgb_copy'+str(i) for i in range(n)]
rgb_copy_clr_name = ['clr_copy'+str(i) for i in range(n)]
points_pred_copy_name = ['pts_pred_copy'+str(i) for i in range(n)]
draw.addTexture(rgb_copy_name, rgb_copy)
draw.addBufferf(points_pred_copy_name, points_pred_copy)
draw.addBufferf(rgb_copy_clr_name, rgb_copy_clr)


# ********************* Display data in loop *********************
color_mode = 0
count = 0
while draw.input():
    if draw.RETURN:
        color_mode = (color_mode + 1) % len(color_dict)
    draw.clear()
    
    draw['rgb'].image(rgb_name[count])
    draw['viz'].image(viz_name[count])
    draw['wld_pred'].size(2).points(points_pred_name[count], color_dict[color_mode][count])
    draw['wld_pred'].object(camera, tex=rgb_name[count])
    draw['wld_pred2'].size(2).points(points_pred_copy_name[count], rgb_copy_clr_name[count])
    draw['wld_pred2'].object(camera, tex=rgb_copy_name[count])

    draw.update(80)
    count = count + 1 if count < (n-1) else 0
