import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def cartesian_to_polar(x, y):
    # 计算半径
    radius = torch.sqrt(x**2 + y**2)

    # 计算角度（弧度）
    theta = torch.atan2(y, x)

    return radius, theta

def polar_to_cartesian(radius, theta):
    # 将极坐标转换为直角坐标
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)

    return x, y

def affine_transform_to_polar(cartesian_points):
    # 提取 x 和 y 坐标
    x, y = cartesian_points[:, 0], cartesian_points[:, 1]

    # 转换为极坐标
    radius, theta = cartesian_to_polar(x, y)

    # 创建仿射变换矩阵
    transformation_matrix = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=torch.float32)

    # 添加 batch 维度
    polar_points = torch.stack([radius, theta], dim=-1).unsqueeze(0)

    # 使用仿射变换矩阵将极坐标点转换为直角坐标点
    transformed_points = F.affine_grid(transformation_matrix.unsqueeze(0), polar_points.size())

    return transformed_points.squeeze(0)

def inverse_affine_transform_to_cartesian(transformed_points):
    # 创建逆变换的仿射变换矩阵
    inverse_transformation_matrix = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=torch.float32)

    # 使用逆仿射变换矩阵将变换后的点转换回直角坐标系
    restored_points = F.affine_grid(inverse_transformation_matrix.unsqueeze(0), transformed_points.size())

    return restored_points.squeeze(0)

# 生成一些直角坐标点
cartesian_points = torch.randn(100, 2)

# 转换为极坐标系下的点
polar_points = affine_transform_to_polar(cartesian_points)

# 将极坐标点逆转为直角坐标系下的点
restored_points = inverse_affine_transform_to_cartesian(polar_points)

# 绘制原始直角坐标点和逆转后的直角坐标点
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(cartesian_points[:, 0], cartesian_points[:, 1], label='Original Cartesian Coordinates')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(restored_points[:, 0], restored_points[:, 1], label='Restored Cartesian Coordinates')
plt.legend()

plt.show()




def get_sample_locations(alpha, phi, dmin, ds, n_azimuth, n_radius, img_size, subdiv, distort ,  fov=0, focal=0,xi=0,  radius_buffer=0, azimuth_buffer=0,img_B=1):
    # import pdb;pdb.set_trace()
    """Get the sample locations in a given radius and azimuth range
    
    Args:
        alpha (array): width of the azimuth range (radians)
        phi (array): phase shift of the azimuth range  (radians)
        dmin (array): minimum radius of the patch (pixels)
        ds (array): distance between the inner and outer arcs of the patch (pixels)
        n_azimuth (int): number of azimuth samples
        n_radius (int): number of radius samples
        img_size (tuple): the size of the image (width, height)
        radius_buffer (int, optional): radius buffer (pixels). Defaults to 0.
        azimuth_buffer (int, optional): azimuth buffer (radians). Defaults to 0.
    
    Returns:
        tuple[ndarray, ndarray]: lists of x and y coordinates of the sample locations
    """
    #Compute center of the image to shift the samples later
    # import pdb;pdb.set_trace()
    new_f = focal
    rad = lambda x: new_f*torch.sin(torch.arctan(x))/(xi + torch.cos(torch.arctan(x))) 
    inverse_rad = lambda r: torch.tan(torch.arcsin(xi*r/(new_f)/torch.sqrt(1 + (r/(new_f))*(r/(new_f)))) + torch.arctan(r/(new_f)))

    center = [img_size[0]/2, img_size[1]/2]
    if img_size[0] % 2 == 0:
        center[0] -= 0.5
    if img_size[1] % 2 == 0:
        center[1] -= 0.5
    # import pdb;pdb.set_trace()
    # Sweep start and end
    r_end = dmin + ds 
    # - radius_buffer
    r_start = dmin 
    # + radius_buffer
    alpha_start = phi 
#    print("alpha_start",alpha_start.shape)
    # B = dmin.shape[1]
    B = img_B

    # + azimuth_buffer
    alpha_end = alpha + phi 
    # - azimuth_buffer
    # import pdb;pdb.set_trace()
    # Get the sample locations
    # import pdb;pdb.set_trace()
    # r1 = linspace(r_start, r_end, n_radius)
#    print("r_start, r_end, n_radius",r_start.shape, r_end.shape, n_radius)# torch.Size([16384, 2]) torch.Size([16384, 2]) 1
    # if distort == 'spherical':
    #     radius = linspace(inverse_rad(r_start), inverse_rad(r_end), n_radius)
    #     radius = rad(radius)
    # elif distort  == 'polynomial':
    #     radius = linspace(r_start, r_end, n_radius)
    # import pdb;pdb.set_trace()
    radius = r_start.unsqueeze(0)
#    print("radius1",radius.shape)#torch.Size([1, 16384, 2])
    radius = torch.transpose(radius, 0,1)
#    print("radius2",radius)
    radius = radius.reshape(radius.shape[0]*radius.shape[1], B)
#    print("radius3",radius)
    # azimuth = linspace(alpha_start, alpha_end, n_azimuth)
    azimuth = alpha_start.unsqueeze(0)
#    print("azimuth1",azimuth)
    azimuth = torch.transpose(azimuth, 0,1)
#    print("azimuth2",azimuth)
    azimuth = azimuth.flatten()
#    print("azimuth3",azimuth)
    azimuth = azimuth.reshape(azimuth.shape[0], 1).repeat_interleave(B, 1)
#    print("azimuth4",azimuth)

    
    azimuth = azimuth.reshape(1, azimuth.shape[0], B).repeat_interleave(n_radius, 0)
#    print("azimuth5",azimuth)
    radius = radius.reshape(radius.shape[0], 1, B).repeat_interleave(n_azimuth, 1)
#    print("radius4",radius)
    # import pdb;pdb.set_trace()
    radius_mesh = radius.reshape(subdiv[0]*subdiv[1], n_radius, n_azimuth, B)
#    print("radius_mesh1",radius_mesh.shape)
    # import pdb;pdb.set_trace()
    # d = radius_mesh[0][0][0][0] - radius_mesh[0][1][0][0]
    # eps = np.random.normal(0, d/3)
    # radius_mesh = random.uniform(radius_mesh-d, radius_mesh+d)
    # radius_mesh = radius_mesh + eps
    # import pdb;pdb.set_trace()
    azimuth_mesh = azimuth.reshape(n_radius, subdiv[0]*subdiv[1], n_azimuth, B).transpose(0,1)  
#    print("azimuth_mesh",azimuth_mesh.shape)
    azimuth_mesh_cos  = torch.cos(azimuth_mesh) 
    azimuth_mesh_sine = torch.sin(azimuth_mesh) 
    x = radius_mesh * azimuth_mesh_cos    # takes time the cosine and multiplication function 
    y = radius_mesh * azimuth_mesh_sine
    
    

    # r = torch.sqrt(x**2+y**2)    # takes time the cosine and multiplication function 
    # theta = torch.arctan(y/x)
    
    # import pdb;pdb.set_trace()
    return x.reshape(subdiv[0]*subdiv[1], n_radius*n_azimuth, B).transpose(1, 2).transpose(0,1), y.reshape(subdiv[0]*subdiv[1], n_radius*n_azimuth, B).transpose(1, 2).transpose(0,1)
