import torch
import torch.nn as nn
import torch.nn.functional as F

import neural_renderer as nr

from losses import LaplacianLoss, LaplacianFaceLoss


class ForwardRenderer(nr.Renderer):
    def __init__(self, sigma_val, sigma_num, sigma_mul, gamma_val, gamma_num, gamma_mul, *args, **kwargs):
        super(ForwardRenderer, self).__init__(*args, **kwargs)

        dim_hidden = [64, 128, 256, 256]

        self.sigma_val = sigma_val
        self.sigma_num = sigma_num
        self.sigma_mul = sigma_mul

        self.gamma_val = gamma_val
        self.gamma_num = gamma_num
        self.gamma_mul = gamma_mul

        # self.conv1 = nn.Conv2d(3 * self.sigma_num * self.gamma_num, dim_hidden[0], kernel_size=5, stride=1, padding=2)
        # self.conv2 = nn.Conv2d(dim_hidden[0], dim_hidden[1], kernel_size=5, stride=1, padding=2)
        # self.conv3 = nn.Conv2d(dim_hidden[1], dim_hidden[2], kernel_size=5, stride=1, padding=2)
        # self.conv4 = nn.Conv2d(dim_hidden[2], dim_hidden[3], kernel_size=5, stride=1, padding=2)
        # self.conv5 = nn.Conv2d(dim_hidden[3], 3, kernel_size=5, stride=1, padding=2)

    def forward(self, vertices, faces, textures=None, mode=None):
        '''
        Implementation of forward rendering method
        The old API is preserved for back-compatibility with the Chainer implementation
        '''

        if mode is None:
            return self.render(vertices, faces, textures)
        elif mode == 'silhouettes':
            return self.render_silhouettes(vertices, faces)
        elif mode == 'depth':
            return self.render_depth(vertices, faces)
        elif mode == 'shading':
            return self.render_shading(vertices, faces, textures)
        else:
            raise ValueError("mode should be one of None, 'silhouettes' or 'depth'")

    def forward_render(self, faces, image_size, image_real=None, sigma_val=.03, sigma_num=5, sigma_mul=3):
        batch_size = faces.size(0)
        num_face = faces.size(1)

        x = nr.transform_triangle(faces, image_size, sigma_val, sigma_num, sigma_mul)  # [nb, nf, is, is] * 3

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = F.relu(self.conv5(x), inplace=True)
        x = torch.sigmoid(self.conv6(x)).squeeze()

        return x

    def forward_eff_render(self, faces, image_size, sigma_val=.01, sigma_num=7, sigma_mul=2):
        batch_size = faces.size(0)
        num_face = faces.size(1)

        x = nr.efficient_transform_triangle(faces, image_size, sigma_val, sigma_num, sigma_mul)  # [nb, nf, is, is]

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = torch.sigmoid(self.conv5(x)).squeeze()

        return x

    def forward_eff_render_shading(self, faces, textures, image_size, sigma_val=.01, sigma_num=7, sigma_mul=2,
                                   gamma_val=.01, gamma_num=5, gamma_mul=2, near=0.1, far=100):
        batch_size = faces.size(0)
        num_face = faces.size(1)

        # import pdb;pdb.set_trace()

        x = nr.efficient_transform_triangle_shading(faces, textures, image_size, sigma_val, sigma_num, sigma_mul,
                                                    gamma_val, gamma_num, gamma_mul, near, far)  # [nb, nf, is, is]

        dis = x

        relu = lambda x: F.leaky_relu(x, inplace=True, negative_slope=0.2)

        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = relu(self.conv3(x))
        x = relu(self.conv4(x))
        # x = self.conv5(x)#.clamp(0, 1)
        x = torch.tanh(self.conv5(x))

        return x, dis

    def render_silhouettes(self, vertices, faces):
        # fill back
        #        if self.fill_back:
        #            faces = torch.cat((faces, faces[:, :, list(reversed(list(range(faces.shape[-1]))))]), dim=1)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            vertices = nr.projection(vertices, self.P, self.dist_coeffs, self.orig_size)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images_real = nr.rasterize_silhouettes(faces, self.image_size, self.anti_aliasing).detach()
        images_fake = self.forward_eff_render(faces, self.image_size, self.sigma_val, self.sigma_num, self.sigma_mul)

        return images_fake, images_real

    def render_shading(self, vertices, faces, textures):
        # lighting
        faces_lighting = nr.vertices_to_faces(vertices, faces)
        textures = nr.lighting(
            faces_lighting,
            textures,
            self.light_intensity_ambient,
            self.light_intensity_directional,
            self.light_color_ambient,
            self.light_color_directional,
            self.light_direction)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            vertices = nr.projection(vertices, self.P, self.dist_coeffs, self.orig_size)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images_real = nr.rasterize_rgba(faces, textures, self.image_size, self.anti_aliasing)
        # import pdb;pdb.set_trace()
        images_real = torch.cat([images_real['rgb'], images_real['alpha'].unsqueeze(1)], dim=1).detach()
        images_fake = self.forward_eff_render_shading(faces, textures, self.image_size, self.sigma_val, self.sigma_num,
                                                      self.sigma_mul, self.gamma_val, self.gamma_num, self.gamma_mul,
                                                      self.near, self.far)

        # import pdb;pdb.set_trace()

        return images_fake, images_real

    def render_color(self, vertices, faces, textures):
        # lighting
        # faces_lighting = nr.vertices_to_faces(vertices, faces)
        # textures = nr.lighting(
        #     faces_lighting,
        #     textures,
        #     self.light_intensity_ambient,
        #     self.light_intensity_directional,
        #     self.light_color_ambient,
        #     self.light_color_directional,
        #     self.light_direction)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            vertices = nr.projection(vertices, self.P, self.dist_coeffs, self.orig_size)

        # rasterization
        # faces_ = nr.vertices_to_faces(vertices, faces)
        # images_fake, dis = self.forward_eff_render_shading(faces_, textures, self.image_size, self.sigma_val, self.sigma_num, self.sigma_mul, self.gamma_val, self.gamma_num, self.gamma_mul, self.near, self.far)

        faces = torch.cat((faces, faces[:, :, [2, 1, 0]]), dim=1).contiguous()
        textures = torch.cat((textures, textures), dim=1).contiguous()
        faces = nr.vertices_to_faces(vertices, faces)
        images_real = nr.rasterize_rgba(faces, textures, self.image_size, self.anti_aliasing)
        # import pdb;pdb.set_trace()
        images_real = torch.cat([images_real['rgb'], images_real['alpha'].unsqueeze(1)], dim=1)  # .detach()

        # import pdb;pdb.set_trace()

        return images_real

    def render(self, vertices, faces, textures):
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(list(range(faces.shape[-1]))))]), dim=1).detach()
            textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)

        # lighting
        faces_lighting = nr.vertices_to_faces(vertices, faces)
        textures = nr.lighting(
            faces_lighting,
            textures,
            self.light_intensity_ambient,
            self.light_intensity_directional,
            self.light_color_ambient,
            self.light_color_directional,
            self.light_direction)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            vertices = nr.projection(vertices, self.P, self.dist_coeffs, self.orig_size)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize(
            faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color)
        return images


class Encoder(nn.Module):
    def __init__(self, dim_in=4, dim_out=512, dim1=64, dim2=1024, im_size=64):
        super(Encoder, self).__init__()
        dim_hidden = [dim1, dim1 * 2, dim1 * 4, dim2, dim2]
        self.conv1 = nn.Conv2d(dim_in, dim_hidden[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(dim_hidden[0], dim_hidden[1], kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(dim_hidden[0], dim_hidden[0], kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(dim_hidden[1], dim_hidden[2], kernel_size=5, stride=2, padding=2)
        self.conv5 = nn.Conv2d(dim_hidden[1], dim_hidden[1], kernel_size=5, stride=1, padding=2)

        self.bn1 = nn.BatchNorm2d(dim_hidden[0])
        self.bn2 = nn.BatchNorm2d(dim_hidden[1])
        self.bn3 = nn.BatchNorm2d(dim_hidden[2])
        self.bn4 = nn.BatchNorm2d(dim_hidden[0])
        self.bn5 = nn.BatchNorm2d(dim_hidden[1])

        self.fc1 = nn.Linear(dim_hidden[2] * (im_size // 8) ** 2, dim_hidden[3])
        self.fc2 = nn.Linear(dim_hidden[3], dim_hidden[4])
        self.fc3 = nn.Linear(dim_hidden[4], dim_out)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn5(self.conv5(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc3(x), inplace=True)
        return x


TEXTURE_SIZE = 1


class EncoderColor(nn.Module):
    def __init__(self, dim_in=4, dim_out=512, dim1=64, dim2=1024, im_size=64, naive=False):
        super(EncoderColor, self).__init__()
        dim_hidden = [dim1, dim1 * 2, dim1 * 4, dim2, dim2]
        self.conv1 = nn.Conv2d(dim_in, dim_hidden[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(dim_hidden[0], dim_hidden[1], kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(dim_hidden[1], dim_hidden[2], kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(dim_hidden[0])
        self.bn2 = nn.BatchNorm2d(dim_hidden[1])
        self.bn3 = nn.BatchNorm2d(dim_hidden[2])
        self.fc1 = nn.Linear(dim_hidden[2] * (im_size // 8) ** 2, dim_hidden[3])
        self.fc2 = nn.Linear(dim_hidden[3], dim_hidden[4])
        self.fc3 = nn.Linear(dim_hidden[4], dim_out)

        self.nc = 10  # num_color

        self.naive = naive

        if self.naive:
            self.fc4 = nn.Linear(dim_out, 1280 * 3)

        else:
            self.fc4 = nn.Linear(dim_out, 1280 * self.nc * TEXTURE_SIZE * TEXTURE_SIZE * TEXTURE_SIZE)
            self.fc_color = nn.Linear(dim_out, 64 * 64 * self.nc)

    def forward(self, x):
        if not self.naive:
            rgb = x[:, :3].view(-1, 1, 1, 1, 1, 3, 1, 64 * 64)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc3(x), inplace=True)

        if self.naive:
            x = torch.sigmoid(self.fc4(x)).view(-1, 1280, TEXTURE_SIZE, TEXTURE_SIZE, TEXTURE_SIZE, 3)
            return x, 0

        else:
            fc_color = torch.softmax(self.fc_color(x).view(-1, 1, 1, 1, 1, 1, self.nc, 64 * 64), dim=7)
            color = (fc_color * rgb).sum(-1)

            x = self.fc4(x).view(-1, 1280, TEXTURE_SIZE, TEXTURE_SIZE, TEXTURE_SIZE, 1, self.nc)
            softmax = torch.softmax(x, dim=6)
            x = (softmax * color).sum(-1)

            entropy_loss = -(softmax * softmax.log()).sum(6).mean()
            return x, entropy_loss


class Decoder(nn.Module):
    def __init__(self, filename_obj, dim_in=512, centroid_scale=0.1, bias_scale=1.0, centroid_lr=0.1, bias_lr=1.0):
        super(Decoder, self).__init__()
        # load .obj
        vertices_base, faces = nr.load_obj(filename_obj)
        # faces = faces[:, list(reversed(list(range(faces.shape[-1]))))]
        self.register_buffer('vertices_base', vertices_base)
        self.register_buffer('faces', faces)

        self.laplacian_loss = LaplacianFaceLoss(vertices_base, faces)

        self.nv = self.vertices_base.size(0)
        self.nf = self.faces.size(0)
        self.centroid_scale = centroid_scale
        self.bias_scale = bias_scale
        self.obj_scale = 0.5

        dim = 1024
        dim_hidden = [dim, dim * 2]
        self.fc1 = nn.Linear(dim_in, dim_hidden[0])
        self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.fc_centroid = nn.Linear(dim_hidden[1], 3)
        self.fc_bias = nn.Linear(dim_hidden[1], self.nv * 3)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        centroid = self.fc_centroid(x) * self.centroid_scale
        bias = self.fc_bias(x) * self.bias_scale
        bias = bias.view(-1, self.nv, 3)

        base = self.vertices_base * self.obj_scale

        sign = torch.sign(base)
        base = torch.abs(base)
        base = torch.log(base / (1 - base))

        centroid = torch.tanh(centroid[:, None, :])
        scale_pos = 1 - centroid
        scale_neg = centroid + 1

        vertices = torch.sigmoid(base + bias) * sign
        vertices = F.relu(vertices) * scale_pos - F.relu(-vertices) * scale_neg
        vertices = vertices + centroid
        vertices = vertices * 0.5
        faces = self.faces[None, :, :].repeat(batch_size, 1, 1)

        return vertices, faces


class Model(nn.Module):
    def __init__(self, filename_obj, args):
        super(Model, self).__init__()

        self.vertices_predicted_a = None
        self.vertices_predicted_b = None

        self.encoder = Encoder()
        self.decoder = Decoder(filename_obj)
        self.renderer = ForwardRenderer(sigma_val=args.sigma_val, sigma_num=args.sigma_num, sigma_mul=args.sigma_mul,
                                        gamma_val=args.gamma_val, gamma_num=args.gamma_num, gamma_mul=args.gamma_mul,
                                        camera_mode='look_at', image_size=64)
        self.encoder_color = EncoderColor()
        self.renderer.image_size = 64
        self.renderer.viewing_angle = 15.
        self.renderer.anti_aliasing = True

    def load_encoder_state_dict(self, state_dict):
        self.encoder.conv1.weight.data = state_dict['module.encoder.conv1.weight']
        self.encoder.conv1.bias.data = state_dict['module.encoder.conv1.bias']
        self.encoder.conv2.weight.data = state_dict['module.encoder.conv2.weight']
        self.encoder.conv2.bias.data = state_dict['module.encoder.conv2.bias']
        self.encoder.conv3.weight.data = state_dict['module.encoder.conv3.weight']
        self.encoder.conv3.bias.data = state_dict['module.encoder.conv3.bias']
        self.encoder.fc1.weight.data = state_dict['module.encoder.fc1.weight']
        self.encoder.fc1.bias.data = state_dict['module.encoder.fc1.bias']
        self.encoder.fc2.weight.data = state_dict['module.encoder.fc2.weight']
        self.encoder.fc2.bias.data = state_dict['module.encoder.fc2.bias']
        self.encoder.fc3.weight.data = state_dict['module.encoder.fc3.weight']
        self.encoder.fc3.bias.data = state_dict['module.encoder.fc3.bias']
        self.encoder.conv4.weight.data = state_dict['module.encoder.conv4.weight']
        self.encoder.conv4.bias.data = state_dict['module.encoder.conv4.bias']
        self.encoder.conv5.weight.data = state_dict['module.encoder.conv5.weight']
        self.encoder.conv5.bias.data = state_dict['module.encoder.conv5.bias']

        self.encoder.bn1.weight.data = state_dict['module.encoder.bn1.weight']
        self.encoder.bn1.bias.data = state_dict['module.encoder.bn1.bias']
        self.encoder.bn1.running_mean.data = state_dict['module.encoder.bn1.running_mean']
        self.encoder.bn1.running_var.data = state_dict['module.encoder.bn1.running_var']
        self.encoder.bn1.num_batches_tracked.data = state_dict['module.encoder.bn1.num_batches_tracked']

        self.encoder.bn2.weight.data = state_dict['module.encoder.bn2.weight']
        self.encoder.bn2.bias.data = state_dict['module.encoder.bn2.bias']
        self.encoder.bn2.running_mean.data = state_dict['module.encoder.bn2.running_mean']
        self.encoder.bn2.running_var.data = state_dict['module.encoder.bn2.running_var']
        self.encoder.bn2.num_batches_tracked.data = state_dict['module.encoder.bn2.num_batches_tracked']

        self.encoder.bn3.weight.data = state_dict['module.encoder.bn3.weight']
        self.encoder.bn3.bias.data = state_dict['module.encoder.bn3.bias']
        self.encoder.bn3.running_mean.data = state_dict['module.encoder.bn3.running_mean']
        self.encoder.bn3.running_var.data = state_dict['module.encoder.bn3.running_var']
        self.encoder.bn3.num_batches_tracked.data = state_dict['module.encoder.bn3.num_batches_tracked']

        self.encoder.bn4.weight.data = state_dict['module.encoder.bn4.weight']
        self.encoder.bn4.bias.data = state_dict['module.encoder.bn4.bias']
        self.encoder.bn4.running_mean.data = state_dict['module.encoder.bn4.running_mean']
        self.encoder.bn4.running_var.data = state_dict['module.encoder.bn4.running_var']
        self.encoder.bn4.num_batches_tracked.data = state_dict['module.encoder.bn4.num_batches_tracked']

        self.encoder.bn5.weight.data = state_dict['module.encoder.bn5.weight']
        self.encoder.bn5.bias.data = state_dict['module.encoder.bn5.bias']
        self.encoder.bn5.running_mean.data = state_dict['module.encoder.bn5.running_mean']
        self.encoder.bn5.running_var.data = state_dict['module.encoder.bn5.running_var']
        self.encoder.bn5.num_batches_tracked.data = state_dict['module.encoder.bn5.num_batches_tracked']

        self.decoder.fc1.weight.data = state_dict['module.decoder.fc1.weight']
        self.decoder.fc1.bias.data = state_dict['module.decoder.fc1.bias']
        self.decoder.fc2.weight.data = state_dict['module.decoder.fc2.weight']
        self.decoder.fc2.bias.data = state_dict['module.decoder.fc2.bias']
        self.decoder.fc_centroid.weight.data = state_dict['module.decoder.fc_centroid.weight']
        self.decoder.fc_centroid.bias.data = state_dict['module.decoder.fc_centroid.bias']
        self.decoder.fc_bias.weight.data = state_dict['module.decoder.fc_bias.weight']
        self.decoder.fc_bias.bias.data = state_dict['module.decoder.fc_bias.bias']

        self.encoder.bn1.eval()
        self.encoder.bn2.eval()
        self.encoder.bn3.eval()
        self.encoder.bn4.eval()
        self.encoder.bn5.eval()

    def renderer_param(self):
        return self.renderer.parameters()

    def model_param(self):
        return list(self.encoder_color.parameters())

    def predict(self, image_a, image_b, viewpoint_a, viewpoint_b):
        batch_size = image_a.size(0)
        images = torch.cat((image_a, image_b), dim=0)
        viewpoints = torch.cat((viewpoint_a, viewpoint_a, viewpoint_b, viewpoint_b), dim=0)
        self.renderer.eye = viewpoints

        vertices, faces = self.decoder(self.encoder(images))

        images = images

        colors, entropy_loss = self.encoder_color(images)

        laplacian_loss = self.decoder.laplacian_loss(colors.sum((2, 3, 4)).squeeze())
        mean_loss = (colors - colors.sum((2, 3, 4), keepdim=True) / TEXTURE_SIZE ** 3).pow(2).sum((2, 3, 4)).mean()

        vertices = torch.cat((vertices, vertices), dim=0)
        faces = torch.cat((faces, faces), dim=0)
        # import pdb;pdb.set_trace()
        textures = torch.cat((colors, colors), dim=0).view(-1, faces.size(1), TEXTURE_SIZE, TEXTURE_SIZE, TEXTURE_SIZE,
                                                           3)

        # textures = torch.ones(4 * batch_size, faces.size(1), 1, 1, 1, 3, dtype=torch.float32).cuda()

        # silhouettes_fake, silhouettes_real = self.renderer.render_silhouettes(vertices, faces)
        silhouettes_real = self.renderer.render_color(vertices, faces, textures)
        # import pdb;pdb.set_trace()

        return silhouettes_real.chunk(4, dim=0), vertices[:2 * batch_size].chunk(2,
                                                                                 dim=0), laplacian_loss, entropy_loss, mean_loss

    def reconstruct(self, images, viewpoints=None, save_path=None, fill_back=False):
        vertices, faces = self.decoder(self.encoder(images))
        textures = self.encoder_color(images)[0].view(-1, faces.size(1), TEXTURE_SIZE, TEXTURE_SIZE, TEXTURE_SIZE, 3)
        textures = textures.permute((0, 1, 4, 3, 2, 5))
        if save_path is not None:
            if not fill_back:
                nr.save_obj(save_path, vertices[0], faces[0])
            else:
                faces = torch.cat((faces, faces[:, :, list(reversed(list(range(faces.shape[-1]))))]), dim=1)
                textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)
                nr.save_obj(save_path, vertices[0], faces[0])
        if viewpoints is None:
            if fill_back:
                faces = torch.cat((faces, faces[:, :, list(reversed(list(range(faces.shape[-1]))))]), dim=1)
                textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)
            return vertices, faces, textures
        else:
            self.renderer.eye = viewpoints
            textures = torch.ones(viewpoints.size(0), faces.size(1), 2, 2, 2, 3).float().cuda()
            images_out = self.renderer.render(vertices, faces, textures)
            return images_out, textures

    def forward(self, images=None, viewpoints=None, task='train'):
        if task == 'train':
            return self.predict(images[0], images[1], viewpoints[0], viewpoints[1])

    def evaluate_iou(self, images, voxels):
        vertices, faces = self.decoder(self.encoder(images))
        textures = self.encoder_color(images)[0].view(-1, faces.size(1), TEXTURE_SIZE, TEXTURE_SIZE, TEXTURE_SIZE, 3)
        faces_ = nr.vertices_to_faces(vertices, faces).data
        faces_norm = faces_ * 1. * (32. - 1) / 32. + 0.5
        voxels_predict = nr.voxelization(faces_norm, 32, False).cpu().numpy()
        voxels_predict = voxels_predict.transpose(0, 2, 1, 3)[:, :, :, ::-1]
        iou = (voxels * voxels_predict).sum((1, 2, 3)) / (0 < (voxels + voxels_predict)).sum((1, 2, 3))
        return iou, vertices, faces, textures

    @property
    def L1_norm(self):
        L1_param = []
        for name, W in self.encoder_color.named_parameters():
            L1 = W.norm(p=1)
            L1_param.append(L1)
        return sum(L1_param)

    def evaluate(self, images):
        vertices, faces = self.decoder(self.encoder(images))
        textures = self.encoder_color(images)[0].view(-1, faces.size(1), TEXTURE_SIZE, TEXTURE_SIZE, TEXTURE_SIZE, 3)
        faces_ = nr.vertices_to_faces(vertices, faces).data
        faces_norm = faces_ * 1. * (32. - 1) / 32. + 0.5
        voxels_predict = nr.voxelization(faces_norm, 32, False).cpu().numpy()
        voxels_predict = voxels_predict.transpose(0, 2, 1, 3)[:, :, :, ::-1]
        return vertices, faces, textures
