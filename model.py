import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
from SD_branch import SD_Module


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)


class VGG(nn.Module):
	def __init__(self, features):
		super(VGG, self).__init__()
		self.features = features
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 1000),
		)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x


class extractor(nn.Module):
	def __init__(self, pretrained):
		super(extractor, self).__init__()
		vgg16_bn = VGG(make_layers(cfg, batch_norm=True))
		if pretrained:
			vgg16_bn.load_state_dict(torch.load('./output/vgg16.pth'))
		self.features = vgg16_bn.features
	
	def forward(self, x):
		out = []
		for m in self.features:
			x = m(x)
			if isinstance(m, nn.MaxPool2d):
				out.append(x)
		return out[1:]

class SpaceToDepth(nn.Module):
	def __init__(self, block_size):
		super(SpaceToDepth, self).__init__()
		self.block_size = block_size
		self.block_size_sq = block_size*block_size
	def forward(self, input):
		output = input.permute(0, 2, 3, 1)
		(batch_size, s_height, s_width, s_depth) = output.size()
		d_depth = s_depth * self.block_size_sq
		d_width = int(s_width / self.block_size)
		d_height = int(s_height / self.block_size)
		t_1 = output.split(self.block_size, 2)
		stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
		output = torch.stack(stack, 1)
		output = output.permute(0, 2, 1, 3)
		output = output.permute(0, 3, 1, 2)
		return output

class SPM(nn.Module):
	def __init__(self, in_channels, out_channels, norm_layer):
		super(SPM, self).__init__()
		inter_channels = in_channels
		self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
									norm_layer(inter_channels),
									nn.ReLU())
		self.sd = SD_Module(inter_channels)
		self.si = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
								norm_layer(inter_channels),
								nn.ReLU())

		self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
									norm_layer(out_channels),
									nn.ReLU())
		self.conv3 = nn.Sequential( nn.Conv2d(inter_channels, out_channels, 1), norm_layer(out_channels),
								   nn.ReLU())

	def forward(self, x):
		feat1 = self.conv1(x)
		sd_feat = self.sd(feat1)
		sd_feat = self.conv2(sd_feat)

		si_feat_att = self.si(x)
		si_feat_att = torch.sigmoid(si_feat_att)
		si_feat = x * si_feat_att + x
		spm_output = sd_feat + si_feat
		spm_output = self.conv3(spm_output)
		return spm_output


class merge(nn.Module):
	def __init__(self):
		super(merge, self).__init__()
		channel_all = 256
		self.head = SPM(512, 512, norm_layer = nn.BatchNorm2d)
		self.conv_trans1 = nn.Conv2d(512, 4096, 1)
		self.bn_trans1 = nn.BatchNorm2d(4096)
		self.relu_trans1 = nn.ReLU()
		self.trans1 = nn.PixelShuffle(upscale_factor=4)
		self.conv1 = nn.Conv2d(1024, 128, 1)
		self.bn1 = nn.BatchNorm2d(128)
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(128)
		self.relu2 = nn.ReLU()
		self.conv_trans2 = nn.Conv2d(128, 1024, 1)
		self.bn_trans2 = nn.BatchNorm2d(1024)
		self.relu_trans2 = nn.ReLU()
		self.trans2 = nn.PixelShuffle(upscale_factor=2)
		self.conv3 = nn.Conv2d(384, 64, 1)
		self.bn3 = nn.BatchNorm2d(64)
		self.relu3 = nn.ReLU()
		self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.relu4 = nn.ReLU()
		self.conv_trans3 = nn.Conv2d(64, 256, 1)
		self.bn_trans3 = nn.BatchNorm2d(256)
		self.relu_trans3 = nn.ReLU()
		self.conv5 = nn.Conv2d(192, 32, 1)
		self.bn5 = nn.BatchNorm2d(32)
		self.relu5 = nn.ReLU()
		self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn6 = nn.BatchNorm2d(32)
		self.relu6 = nn.ReLU()
		self.conv_trans4 = nn.Conv2d(128, 256, 1)
		self.bn_trans4 = nn.BatchNorm2d(256)
		self.relu_trans4 = nn.ReLU()
		#ATT sequential
		self.fg_att_1 = nn.Conv2d(channel_all, channel_all, 3, padding=1)
		self.bn_fg_att_1 = nn.BatchNorm2d(channel_all)
		self.relu_fg_att_1 = nn.Sigmoid()
		self.fg_att_2 = nn.Conv2d(channel_all, channel_all, 3, padding=1)
		self.bn_fg_att_2 = nn.BatchNorm2d(channel_all)
		self.relu_fg_att_2 = nn.Sigmoid()
		self.fg_att_3 = nn.Conv2d(channel_all, channel_all, 3, padding=1)
		self.bn_fg_att_3 = nn.BatchNorm2d(channel_all)
		self.relu_fg_att_3 = nn.Sigmoid()
		#sequential  conv
		self.fg_1 = nn.Conv2d(channel_all, channel_all, 3, padding=1)
		self.bn_fg_1 = nn.BatchNorm2d(channel_all)
		self.relu_fg_1 = nn.ReLU()
		self.fg_1_1 = nn.Conv2d(channel_all, channel_all, 3, padding=1)
		self.bn_fg_1_1 = nn.BatchNorm2d(channel_all)
		self.relu_fg_1_1 = nn.ReLU()
		self.fg_2 = nn.Conv2d(channel_all, channel_all, 3, padding=1)
		self.bn_fg_2 = nn.BatchNorm2d(channel_all)
		self.relu_fg_2 = nn.ReLU()
		self.fg_concate_2 = nn.Conv2d(2*channel_all, channel_all, 3, padding=1)
		self.bn_fg_concate_2 = nn.BatchNorm2d(channel_all)
		self.relu_fg_concate_2 = nn.ReLU()
		self.fg_3 = nn.Conv2d(channel_all, channel_all, 3, padding=1)
		self.bn_fg_3 = nn.BatchNorm2d(channel_all)
		self.relu_fg_3 = nn.ReLU()
		self.fg_concate_3 = nn.Conv2d(channel_all*2, channel_all, 3, padding=1)
		self.bn_fg_concate_3 = nn.BatchNorm2d(channel_all)
		self.relu_fg_concate_3 = nn.ReLU()
		self.fg_4 = nn.Conv2d(channel_all, channel_all, 3, padding=1)
		self.bn_fg_4 = nn.BatchNorm2d(channel_all)
		self.relu_fg_4 = nn.ReLU()
		#att reversed order
		self.fh_att_1 = nn.Conv2d(channel_all, channel_all, 3, padding=1)
		self.bn_fh_att_1 = nn.BatchNorm2d(channel_all)
		self.relu_fh_att_1 = nn.Sigmoid()
		self.fh_att_2 = nn.Conv2d(channel_all, channel_all, 3, padding=1)
		self.bn_fh_att_2 = nn.BatchNorm2d(channel_all)
		self.relu_fh_att_2 = nn.Sigmoid()
		self.fh_att_3 = nn.Conv2d(channel_all, channel_all, 3, padding=1)
		self.bn_fh_att_3 = nn.BatchNorm2d(channel_all)
		self.relu_fh_att_3 = nn.Sigmoid()
		#reversed order  conv
		self.fh_4 = nn.Conv2d(channel_all, channel_all, 3, padding=1)
		self.bn_fh_4 = nn.BatchNorm2d(channel_all)
		self.relu_fh_4 = nn.ReLU()
		self.fh_4_4 = nn.Conv2d(channel_all, channel_all, 3, padding=1)
		self.bn_fh_4_4 = nn.BatchNorm2d(channel_all)
		self.relu_fh_4_4 = nn.ReLU()
		self.fh_3 = nn.Conv2d(channel_all, channel_all, 3, padding=1)
		self.bn_fh_3 = nn.BatchNorm2d(channel_all)
		self.relu_fh_3 = nn.ReLU()
		self.fh_concate_3 = nn.Conv2d(2*channel_all, channel_all, 3, padding=1)
		self.bn_fh_concate_3 = nn.BatchNorm2d(channel_all)
		self.relu_fh_concate_3 = nn.ReLU()
		self.fh_2 = nn.Conv2d(channel_all, channel_all, 3, padding=1)
		self.bn_fh_2 = nn.BatchNorm2d(channel_all)
		self.relu_fh_2 = nn.ReLU()
		self.fh_concate_2 = nn.Conv2d(channel_all*2, channel_all, 3, padding=1)
		self.bn_fh_concate_2 = nn.BatchNorm2d(channel_all)
		self.relu_fh_concate_2 = nn.ReLU()
		self.fh_1 = nn.Conv2d(channel_all, channel_all, 3, padding=1)
		self.bn_fh_1 = nn.BatchNorm2d(channel_all)
		self.relu_fh_1 = nn.ReLU()
		self.fh_concate_1 = nn.Conv2d(channel_all*2, channel_all, 3, padding=1)
		self.bn_fh_concate_1 = nn.BatchNorm2d(channel_all)
		self.relu_fh_concate_1 = nn.ReLU()
		# concate part
		self.concate_conv1 = nn.Conv2d(channel_all*3, channel_all, 3, padding=1)
		self.bn_concate_1 = nn.BatchNorm2d(channel_all)
		self.relu_concate_1 = nn.ReLU()
		self.concate_conv2 = nn.Conv2d(channel_all*4, channel_all, 3, padding=1)
		self.bn_concate_2 = nn.BatchNorm2d(channel_all)
		self.relu_concate_2 = nn.ReLU()
		self.concate_conv3 = nn.Conv2d(channel_all*4, channel_all, 3, padding=1)
		self.bn_concate_3 = nn.BatchNorm2d(channel_all)
		self.relu_concate_3 = nn.ReLU()
		self.concate_conv4 = nn.Conv2d(channel_all*3, channel_all, 3, padding=1)
		self.bn_concate_4 = nn.BatchNorm2d(channel_all)
		self.relu_concate_4 = nn.ReLU()
		self.concate_total = nn.Conv2d(4*channel_all, channel_all, 3, padding=1)
		self.bn_concate_total = nn.BatchNorm2d(channel_all)
		self.relu_concate_total = nn.ReLU()
		#  output
		self.conv7 = nn.Conv2d(384, 64, 3, padding=1)
		self.bn7 = nn.BatchNorm2d(64)
		self.relu7 = nn.ReLU()
		self.conv8 = nn.Conv2d(64, 32, 3, padding=1)
		self.bn8 = nn.BatchNorm2d(32)
		self.relu8 = nn.ReLU()
		self.conv9 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn9 = nn.BatchNorm2d(32)
		self.relu9 = nn.ReLU()
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		res_att_2 = self.head(x[3])
		f1 = res_att_2
		f1 = self.relu_trans1(self.bn_trans1(self.conv_trans1(f1)))
		f1 = self.trans1(f1)
		f2_temp = F.interpolate(res_att_2, scale_factor=2, mode='bilinear', align_corners=True)
		f2_temp = torch.cat((f2_temp, x[2]), 1)
		f2_temp = self.relu1(self.bn1(self.conv1(f2_temp)))
		f2_temp = self.relu2(self.bn2(self.conv2(f2_temp)))
		f2 = self.relu_trans2(self.bn_trans2(self.conv_trans2(f2_temp)))
		f2 = self.trans2(f2)
		f3_temp = F.interpolate(f2_temp, scale_factor=2, mode='bilinear', align_corners=True)
		f3_temp = torch.cat((f3_temp, x[1]), 1)
		f3_temp = self.relu3(self.bn3(self.conv3(f3_temp)))
		f3_temp = self.relu4(self.bn4(self.conv4(f3_temp)))
		f3 = self.relu_trans3(self.bn_trans3(self.conv_trans3(f3_temp)))
		f4_temp = F.interpolate(f3_temp, scale_factor=2, mode='bilinear', align_corners=True)
		f4_temp = torch.cat((f4_temp, x[0]), 1)
		f4_temp = self.relu5(self.bn5(self.conv5(f4_temp)))
		f4_temp = self.relu6(self.bn6(self.conv6(f4_temp)))
		f4_1 = f4_temp[:,:, 0::2,0::2]
		f4_2 = f4_temp[:,:, 0::2,1::2]
		f4_3 = f4_temp[:,:, 1::2,0::2]
		f4_4 = f4_temp[:,:, 1::2,1::2]
		f4 =  torch.cat((f4_1, f4_2, f4_3, f4_4), 1)
		f4 = self.relu_trans4(self.bn_trans4(self.conv_trans4(f4)))
		# sequential conv
		fg1_att = self.relu_fg_att_1(self.bn_fg_att_1(self.fg_att_1(f1)))
		fg2_att = self.relu_fg_att_2(self.bn_fg_att_2(self.fg_att_2(f2)))
		fg3_att = self.relu_fg_att_3(self.bn_fg_att_3(self.fg_att_3(f3)))
		g1_temp = self.relu_fg_1(self.bn_fg_1(self.fg_1(f1)))
		g1 = self.relu_fg_1_1(self.bn_fg_1_1(self.fg_1_1(g1_temp)))
		g1 = fg1_att * g1 + g1
		g2_temp = self.relu_fg_2(self.bn_fg_2(self.fg_2(f2)))
		g2_temp = torch.cat((g2_temp, g1), 1)
		g2 = self.relu_fg_concate_2(self.bn_fg_concate_2(self.fg_concate_2(g2_temp)))
		g2 = fg2_att * g2 + g2
		g3_temp = self.relu_fg_3(self.bn_fg_3(self.fg_3(f3)))
		g3_temp = torch.cat((g3_temp, g2), 1)
		g3 = self.relu_fg_concate_3(self.bn_fg_concate_3(self.fg_concate_3(g3_temp)))
		g3 = fg3_att * g3 + g3
		g4_temp = self.relu_fg_4(self.bn_fg_4(self.fg_4(f4)))
		g4 = torch.cat((g4_temp, g3), 1)

		#reversed order
		fh1_att = self.relu_fh_att_1(self.bn_fh_att_1(self.fh_att_1(f2)))
		fh2_att = self.relu_fh_att_2(self.bn_fh_att_2(self.fh_att_2(f3)))
		fh3_att = self.relu_fh_att_3(self.bn_fh_att_3(self.fh_att_3(f4)))
		h4_temp = self.relu_fh_4(self.bn_fh_4(self.fh_4(f4)))
		h4 = self.relu_fh_4_4(self.bn_fh_4_4(self.fh_4_4(h4_temp)))
		h4 = fh3_att * h4 + h4
		h3_temp = self.relu_fh_3(self.bn_fh_3(self.fh_3(f3)))
		h3_temp = torch.cat((h3_temp, h4), 1)
		h3 = self.relu_fh_concate_3(self.bn_fh_concate_3(self.fh_concate_3(h3_temp)))
		h3 = fh2_att * h3 + h3
		h2_temp = self.relu_fh_2(self.bn_fh_2(self.fh_2(f2)))
		h2_temp = torch.cat((h2_temp, h3), 1)
		h2 = self.relu_fh_concate_2(self.bn_fh_concate_2(self.fh_concate_2(h2_temp)))
		h2 = fh1_att * h2 + h2
		h1_temp = self.relu_fh_1(self.bn_fh_1(self.fh_1(f1)))
		h1 = torch.cat((h1_temp, h2), 1)
		#   concate
		final_1 = self.relu_concate_1(self.bn_concate_1(self.concate_conv1(torch.cat((g1_temp, h1), 1))))
		final_2 = self.relu_concate_2(self.bn_concate_2(self.concate_conv2(torch.cat((g2_temp, h2_temp), 1))))
		final_3 = self.relu_concate_3(self.bn_concate_3(self.concate_conv3(torch.cat((g3_temp, h3_temp), 1))))
		final_4 = self.relu_concate_4(self.bn_concate_4(self.concate_conv4(torch.cat((g4, h4_temp), 1))))
		final_total = self.relu_concate_total(self.bn_concate_total(self.concate_total(torch.cat((final_1, final_2, final_3, final_4), 1))))
		final_total = F.interpolate(final_total, scale_factor=2, mode='bilinear', align_corners=True)
		final_total = self.relu7(self.bn7(self.conv7(torch.cat((final_total, x[0]), 1))))
		final_total = self.relu8(self.bn8(self.conv8(final_total)))
		final_total = self.relu9(self.bn9(self.conv9(final_total)))
		return final_total

class output(nn.Module):
	def __init__(self, scope=512):
		super(output, self).__init__()
		self.conv1 = nn.Conv2d(32, 1, 1)
		self.sigmoid1 = nn.Sigmoid()
		self.conv2 = nn.Conv2d(32, 4, 1)
		self.sigmoid2 = nn.Sigmoid()
		self.conv3 = nn.Conv2d(32, 1, 1)
		self.sigmoid3 = nn.Sigmoid()
		self.scope = 512
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def forward(self, x):
		sc_map = self.sigmoid1(self.conv1(x))
		regression   = self.sigmoid2(self.conv2(x)) * self.scope
		angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi
		reg_final   = torch.cat((regression, angle), 1)
		return sc_map, reg_final
		
	
class R_Net(nn.Module):
	def __init__(self, pretrained=True):
		super(R_Net, self).__init__()
		self.extractor = extractor(pretrained)
		self.merge = merge()
		self.output = output()
	
	def forward(self, x):
		x = self.extractor(x)
		x = self.merge(x)
		output_final = self.output(x)
		return output_final

if __name__ == '__main__':
	model = R_Net()
	x = torch.randn(1, 3, 256, 256)
	score, geo = model(x)
