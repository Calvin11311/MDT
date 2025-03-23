import torch
import numpy as np
from torch._six import inf

def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def linspace(start, stop, num):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32) / (num - 1)
    
    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1).cuda()
    
    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None].cuda()
    
    return out


def get_sample_params_from_subdiv(subdiv, n_radius, n_azimuth, distortion_model, img_size, D, radius_buffer=0, azimuth_buffer=0):
    """Generate the required parameters to sample every patch based on the subdivison
    Args:
        subdiv (tuple[int, int]): the number of subdivisions for which we need to create the 
                                  samples. The format is (radius_subdiv, azimuth_subdiv)
        n_radius (int): number of radius samples
        n_azimuth (int): number of azimuth samples
        img_size (tuple): the size of the image
    Returns:
        list[dict]: the list of parameters to sample every patch
    """
    # import pdb;pdb.set_trace()
    max_radius = min(img_size)/2
    # print("max_radius",max_radius)
    width = img_size[1]
    # D_min = get_inverse_distortion(subdiv[0], D, max_radius)
    if distortion_model == 'spherical': # in case of spherical distortion pass the 
        # import pdb;pdb.set_trace()
        fov = D[2][0]
        f  = D[1]
        xi = D[0]
        D_min, theta_max = get_inverse_dist_spherical(subdiv[0], xi, fov, f)
    elif distortion_model == 'polynomial':
        # import pdb;pdb.set_trace()
        # print("max_radius",max_radius)
        D_min, theta_max = get_inverse_distortion(subdiv[0], D, max_radius)
        # print(type(subdiv[0]), type(D), type(max_radius))
        # print("D_min",D_min.type())
        # print("D_min",len(D_min),len(D_min[0]))
        #subdiv: radius_cuts, self.azimuth_cuts:径向划分
    
    
    # import pdb;pdb.set_trace()
    # D_min = np.array(dmin_list)  ## del
    D_s = torch.diff(D_min, axis = 0)
    # D_s = np.array(ds_list)
    alpha = 2*torch.tensor(np.pi).cuda() / subdiv[1]#角度步长
    # import pdb;pdb.set_trace()

    D_min = D_min[:-1].reshape(1, subdiv[0], D.shape[1]).repeat_interleave(subdiv[1], 0).reshape(subdiv[0]*subdiv[1], D.shape[1])
    D_s = D_s.reshape(1, subdiv[0], D.shape[1]).repeat_interleave(subdiv[1], 0).reshape(subdiv[0]*subdiv[1], D.shape[1])
    phi_start = 0
    phi_end = 2*torch.tensor(np.pi)
    phi_step = alpha
    phi_list = torch.arange(phi_start, phi_end, phi_step)
    p = phi_list.reshape(1, subdiv[1]).repeat_interleave(subdiv[0], 0)
    phi = p.transpose(1,0).flatten().cuda()
    alpha = alpha.repeat_interleave(subdiv[0]*subdiv[1])
    # Generate parameters for each patch
    # import pdb;pdb.set_trace()
    
    if distortion_model == 'spherical':
        params = {
            'alpha': alpha, "phi": phi, "dmin": D_min, "ds": D_s, "n_azimuth": n_azimuth, "n_radius": n_radius,
            "img_size": img_size, "radius_buffer": radius_buffer, "azimuth_buffer": azimuth_buffer, "subdiv" : subdiv, "fov": fov, "xi": xi, "focal" : f, "distort" : distortion_model,
        }
    elif distortion_model == 'polynomial':
        params = {
            'alpha': alpha, "phi": phi, "dmin": D_min, "ds": D_s, "n_azimuth": n_azimuth, "n_radius": n_radius,
            "img_size": img_size, "radius_buffer": radius_buffer, "azimuth_buffer": azimuth_buffer, "subdiv" : subdiv, "distort" : distortion_model,
        }
    # import pdb;pdb.set_trace()

    return params, D_s.reshape(subdiv[1], subdiv[0], D.shape[1]).T, theta_max


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
    # print("r_start",r_start.shape,r_start)
    # + radius_buffer
    alpha_start = phi 
    # print("alpha_start",alpha_start.shape,alpha_start)
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
    # print("radius1",radius.shape)#torch.Size([1, 16384, 2])
    radius = torch.transpose(radius, 0,1)
#   print("radius2",radius)
    radius = radius.reshape(radius.shape[0]*radius.shape[1], B)
#   print("radius3",radius)
#   azimuth = linspace(alpha_start, alpha_end, n_azimuth)
    azimuth = alpha_start.unsqueeze(0)
    # print("azimuth1",azimuth)
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
    # print("xxxx",x)
    # print('yyyy',y)
    

    # r = torch.sqrt(x**2+y**2)    # takes time the cosine and multiplication function 
    # theta = torch.arctan(y/x)
    
    # import pdb;pdb.set_trace()
    return x.reshape(subdiv[0]*subdiv[1], n_radius*n_azimuth, B).transpose(1, 2).transpose(0,1), y.reshape(subdiv[0]*subdiv[1], n_radius*n_azimuth, B).transpose(1, 2).transpose(0,1)


def get_inverse_distortion(num_points, D, max_radius):#subdiv[0]（r_cut）, D, max_radius
    # # import pdb;pdb.set_trace()
    # print("DDD",D)
    # print("D.shape",D.shape)#D.shape torch.Size([4, 1])
    dist_func = lambda x: x.reshape(1, x.shape[0]).repeat_interleave(D.shape[1], 0).flatten() * (1 + torch.outer(D[0], x**2).flatten() + torch.outer(D[1], x**4).flatten() + torch.outer(D[2], x**6).flatten() +torch.outer(D[3], x**8).flatten())
    
    theta_max = dist_func(torch.tensor([1]).cuda())
    
    # print("theta_max",theta_max.shape)
    # import pdb;pdb.set_trace()
    theta = linspace(torch.tensor([0]).cuda(), theta_max, num_points+1).cuda()

    test_radius = torch.linspace(0, 1, 50).cuda()
    test_theta = dist_func(test_radius).reshape(D.shape[1], 50).transpose(1,0)

    radius_list = torch.zeros(num_points*D.shape[1]).reshape(num_points, D.shape[1]).cuda()
    # import pdb;pdb.set_trace()
    for i in range(D.shape[1]):
        for j in range(num_points):
            lower_idx = test_theta[:, i][test_theta[:, i] <= theta[:, i][j]].argmax()
            upper_idx = lower_idx + 1

            x_0, x_1 = test_radius[lower_idx], test_radius[upper_idx]
            y_0, y_1 = test_theta[:, i][lower_idx], test_theta[:, i][upper_idx]

            radius_list[:, i][j] = x_0 + (theta[:, i][j] - y_0) * (x_1 - x_0) / (y_1 - y_0)
    
    # import pdb;pdb.set_trace()
    max_rad = torch.tensor([1]*D.shape[1]).reshape(1, D.shape[1]).cuda()
    return torch.cat((radius_list, max_rad), axis=0)*max_radius, theta_max

def get_inverse_dist_spherical(num_points, xi, fov, new_f):
    # import pdb;pdb.set_trace()
    # xi = torch.tensor(xi).cuda()
    # width = torch.tensor(width).cuda()
    # # focal_length = torch.tensor(focal_length).cuda()
    # fov = compute_fov(focal_length, 0, width)
    # new_xi = xi
    # new_f = compute_focal(fov, new_xi, width)
    # import pdb;pdb.set_trace()
    rad = lambda x: new_f*torch.sin(torch.arctan(x))/(xi + torch.cos(torch.arctan(x))) 
    # rad_1 = lambda x: new_f/8*torch.sin(torch.arctan(x))/(xi + torch.cos(torch.arctan(x))) 
    inverse_rad = lambda r: torch.tan(torch.arcsin(xi*r/(new_f)*(1 + (r/(new_f))*(r/(new_f)))) + torch.arctan(r/(new_f)))
#     theta_d_max = inverse_rad(new_f)
    min = inverse_rad(2.0)
    theta_d_max = torch.tan(fov/2).cuda()
    theta_d = linspace(torch.tensor([0]).cuda(), theta_d_max, num_points+1).cuda()
    t1 = inverse_rad(2.0)
    t2 = inverse_rad(4.0)
    # theta_d_num = linspace(torch.tensor([0]).cuda(), theta_d_max, (num_points+1)*8).cuda()
    theta_d_num1 = linspace(t1, t2, 10).cuda()
    r_list = rad(theta_d)   
    # r_lin = rad(theta_d_num)
    # r_d = rad(theta_d_num1)
    # import pdb;pdb.set_trace()
    return r_list, theta_d_max


def get_sample_locations_reverse(H, W, n_azimuth, n_radius, subdiv,D,img_B=1):
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
    # new_f = focal
    # rad = lambda x: new_f*torch.sin(torch.arctan(x))/(xi + torch.cos(torch.arctan(x))) 
    # inverse_rad = lambda r: torch.tan(torch.arcsin(xi*r/(new_f)/torch.sqrt(1 + (r/(new_f))*(r/(new_f)))) + torch.arctan(r/(new_f)))

    # center = [img_size[0]/2, img_size[1]/2]
    # if img_size[0] % 2 == 0:
    #     center[0] -= 0.5
    # if img_size[1] % 2 == 0:
    #     center[1] -= 0.5
    # import pdb;pdb.set_trace()
    # Sweep start and end
    # r_end = dmin + ds 
    # # - radius_buffer
    # r_start = dmin 
    # # + radius_buffer
    # alpha_start = phi 
    # B = dmin.shape[1]
    B = img_B

    # + azimuth_buffer
    # alpha_end = alpha + phi 
    # - azimuth_buffer
    # import pdb;pdb.set_trace()
    # Get the sample locations
    # import pdb;pdb.set_trace()
    # r1 = linspace(r_start, r_end, n_radius)

    # H = linspace(0, H, n_radius)
    # print("H,W",H,W)
    # print(type(subdiv[0]), type(D), type(H))
    x,theta=get_inverse_distortion(subdiv[0], D, float(H))
    x = x[:-1].reshape(1, subdiv[0], D.shape[1]).repeat_interleave(subdiv[1], 0).reshape(subdiv[0]*subdiv[1], D.shape[1])
    # x = x[:-1].reshape(1, subdiv[0], D.shape[1]).repeat_interleave
    y,theta=get_inverse_distortion(subdiv[0], D, float(W))
    y = y[:-1].reshape(1, subdiv[0], D.shape[1]).repeat_interleave(subdiv[1], 0).reshape(subdiv[0]*subdiv[1], D.shape[1])
    # print("x.shape",x.shape)#2 129 1
    # print("y.shape",y.shape)#2 129 1
    # import pdb;pdb.set_trace()
    x = torch.transpose(x, 0,1)
    x = x.reshape(x.shape[0]*x.shape[1], B)
    # W = linspace(alpha_start, alpha_end, n_azimuth)
    y = torch.transpose(y, 0,1)
    y = y.reshape(x.shape[0]*x.shape[1], B)

    
    y = y.reshape(1, y.shape[0], B).repeat_interleave(n_radius, 0)
    x = x.reshape(x.shape[0], 1, B).repeat_interleave(n_azimuth, 1)
    # import pdb;pdb.set_trace()
    x = x.reshape(H*W, n_radius, n_azimuth, B)
    # import pdb;pdb.set_trace()
    # d = radius_mesh[0][0][0][0] - radius_mesh[0][1][0][0]
    # eps = np.random.normal(0, d/3)
    # radius_mesh = random.uniform(radius_mesh-d, radius_mesh+d)
    # radius_mesh = radius_mesh + eps
    # import pdb;pdb.set_trace()
    y = y.reshape(n_radius, H*W, n_azimuth, B).transpose(0,1)  
    # azimuth_mesh_cos  = torch.cos(azimuth_mesh) 
    # azimuth_mesh_sine = torch.sin(azimuth_mesh) 
    # x = radius_mesh * azimuth_mesh_cos    # takes time the cosine and multiplication function 
    # y = radius_mesh * azimuth_mesh_sine
    
    

    radius = torch.sqrt(x**2+y**2)    # takes time the cosine and multiplication function 
    azimuth = torch.atan2(y, x)
    
    # import pdb;pdb.set_trace()
    return radius.reshape(H*W, n_radius*n_azimuth, B).transpose(1, 2).transpose(0,1), azimuth.reshape(H*W, n_radius*n_azimuth, B).transpose(1, 2).transpose(0,1)
