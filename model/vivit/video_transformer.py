from einops import rearrange, reduce, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from . import utils
import argparse
from .transformer import PatchEmbed, TransformerContainer, get_sine_cosine_pos_emb
from .weight_init import (trunc_normal_, init_from_vit_pretrain_,
	init_from_mae_pretrain_, init_from_kinetics_pretrain_)

import math
from functools import partial


class TimeSformer(nn.Module):
	"""TimeSformer. A PyTorch impl of `Is Space-Time Attention All You Need for
	Video Understanding? <https://arxiv.org/abs/2102.05095>`_

	Args:
		num_frames (int): Number of frames in the video.
		img_size (int | tuple): Size of input image.
		patch_size (int): Size of one patch.
		pretrained (str | None): Name of pretrained model. Default: None.
		embed_dims (int): Dimensions of embedding. Defaults to 768.
		num_heads (int): Number of parallel attention heads in
			TransformerCoder. Defaults to 12.
		num_transformer_layers (int): Number of transformer layers. Defaults to
			12.
		in_channels (int): Channel num of input features. Defaults to 3.
		dropout_p (float): Probability of dropout layer. Defaults to 0.
		conv_type (str): Type of the convolution in PatchEmbed layer. Defaults to Conv2d.
		attention_type (str): Type of attentions in TransformerCoder. Choices
			are 'divided_space_time', 'space_only' and 'joint_space_time'.
			Defaults to 'divided_space_time'.
		norm_layer (dict): Config for norm layers. Defaults to nn.LayerNorm.
		copy_strategy (str): Copy or Initial to zero towards the new additional layer.
		use_learnable_pos_emb (bool): Whether to use learnable position embeddings.
		return_cls_token (bool): Whether to use cls_token to predict class label.
	"""
	supported_attention_types = [
		'divided_space_time', 'space_only', 'joint_space_time'
	]

	def __init__(self,
				 num_frames,
				 img_size=224,
				 patch_size=16,
				 pretrain_pth=None,
				 weights_from='imagenet',
				 embed_dims=768,
				 num_heads=12,
				 num_transformer_layers=12,
				 in_channels=3,
				 conv_type='Conv2d',
				 dropout_p=0.,
				 attention_type='divided_space_time',
				 norm_layer=nn.LayerNorm,
				 copy_strategy='repeat',
				 use_learnable_pos_emb=True,
				 return_cls_token=True,
				 **kwargs):
		super().__init__()
		assert attention_type in self.supported_attention_types, (
			f'Unsupported Attention Type {attention_type}!')

		self.num_frames = num_frames
		self.pretrain_pth = pretrain_pth
		self.weights_from = weights_from
		self.embed_dims = embed_dims
		self.num_transformer_layers = num_transformer_layers
		self.attention_type = attention_type
		self.copy_strategy = copy_strategy
		self.conv_type = conv_type
		self.use_learnable_pos_emb = use_learnable_pos_emb
		self.return_cls_token = return_cls_token

		#tokenize & position embedding
		self.patch_embed = PatchEmbed(
			img_size=img_size,
			patch_size=patch_size,
			in_channels=in_channels,
			embed_dims=embed_dims,
			conv_type=conv_type)
		num_patches = self.patch_embed.num_patches
		
		if self.attention_type == 'divided_space_time':
			# Divided Space Time Attention
			operator_order = ['time_attn','space_attn','ffn']
			container = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=operator_order)

			transformer_layers = container
		else:
			# Sapce Only & Joint Space Time Attention
			operator_order = ['self_attn','ffn']
			container = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=operator_order)

			transformer_layers = container

		self.transformer_layers = transformer_layers
		self.norm = norm_layer(embed_dims, eps=1e-6)
		
		self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dims))
		# whether to add one cls_token in temporal pos_emb
		self.use_cls_token_temporal = operator_order[-2] == 'time_attn'
		if self.use_cls_token_temporal:
			num_frames = num_frames + 1
		else:
			num_patches = num_patches + 1

		# spatial pos_emb
		if use_learnable_pos_emb:
			self.pos_embed = nn.Parameter(torch.zeros(1,num_patches,embed_dims))
		else:
			self.pos_embed = get_sine_cosine_pos_emb(num_patches,embed_dims)
		self.drop_after_pos = nn.Dropout(p=dropout_p)
		
		# temporal pos_emb
		if self.attention_type != 'space_only':	
			if use_learnable_pos_emb:
				self.time_embed = nn.Parameter(torch.zeros(1,num_frames,embed_dims))
			else:
				self.time_embed = get_sine_cosine_pos_emb(num_frames,embed_dims)
			self.drop_after_time = nn.Dropout(p=dropout_p)

		self.init_weights()

	def init_weights(self):
		if self.use_learnable_pos_emb:
			#trunc_normal_(self.pos_embed, std=.02)
			nn.init.trunc_normal_(self.pos_embed, std=.02)
			if self.attention_type != 'space_only':
				nn.init.trunc_normal_(self.time_embed, std=.02)
		trunc_normal_(self.cls_token, std=.02)
		
		if self.pretrain_pth is not None:
			if self.weights_from == 'imagenet':
				init_from_vit_pretrain_(self,
										self.pretrain_pth,
										self.conv_type,
										self.attention_type,
										self.copy_strategy)
			elif self.weights_from == 'kinetics':
				init_from_kinetics_pretrain_(self,
											 self.pretrain_pth)
			else:
				raise TypeError(f'not support the pretrained weight {self.pretrain_pth}')

	@torch.jit.ignore
	def no_weight_decay_keywords(self):
		return {'pos_embed', 'cls_token', 'mask_token'}
	
	def interpolate_pos_encoding(self, x, w, h):
		npatch = x.shape[1] - 1
		N = self.pos_embed.shape[1] - 1
		if npatch == N and w == h:
			return self.pos_embed
		class_pos_embed = self.pos_embed[:, 0]
		patch_pos_embed = self.pos_embed[:, 1:]
		dim = x.shape[-1]
		w0 = w // self.patch_embed.patch_size[0]
		h0 = h // self.patch_embed.patch_size[0]
		# we add a small number to avoid floating point error in the interpolation
		# see discussion at https://github.com/facebookresearch/dino/issues/8
		w0, h0 = w0 + 0.1, h0 + 0.1
		patch_pos_embed = nn.functional.interpolate(
			patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
			scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
			mode='bicubic',
		)
		assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
		patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
		return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
	
	def prepare_tokens(self, x):
		#Tokenize
		b, t, c, h, w = x.shape
		x = self.patch_embed(x)
		
		# Add Position Embedding
		cls_tokens = repeat(self.cls_token, 'b ... -> (repeat b) ...', repeat=x.shape[0])
		if self.use_cls_token_temporal:
			if self.use_learnable_pos_emb:
				x = x + self.pos_embed
			else:
				x = x + self.pos_embed.type_as(x).detach()
			x = torch.cat((cls_tokens, x), dim=1)
		else:
			x = torch.cat((cls_tokens, x), dim=1)
			if self.use_learnable_pos_emb:
				x = x + self.interpolate_pos_encoding(x, w, h) #self.pos_embed
			else:
				x = x + self.interpolate_pos_encoding(x, w, h).type_as(x).detach() #self.pos_embed
		x = self.drop_after_pos(x)

		# Add Time Embedding
		if self.attention_type != 'space_only':
			cls_tokens = x[:b, 0, :].unsqueeze(1)
			if self.use_cls_token_temporal:
				x = rearrange(x[:, 1:, :], '(b t) p d -> (b p) t d', b=b)
				cls_tokens = repeat(cls_tokens, 
									'b ... -> (repeat b) ...',
									repeat=x.shape[0]//b)
				x = torch.cat((cls_tokens, x), dim=1)
				if self.use_learnable_pos_emb:
					x = x + self.time_embed
				else:
					x = x + self.time_embed.type_as(x).detach()
				cls_tokens = x[:b, 0, :].unsqueeze(1)
				x = rearrange(x[:, 1:, :], '(b p) t d -> b (p t) d', b=b)
				x = torch.cat((cls_tokens, x), dim=1)
			else:
				x = rearrange(x[:, 1:, :], '(b t) p d -> (b p) t d', b=b)
				if self.use_learnable_pos_emb:
					x = x + self.time_embed
				else:
					x = x + self.time_embed.type_as(x).detach()
				x = rearrange(x, '(b p) t d -> b (p t) d', b=b)
				x = torch.cat((cls_tokens, x), dim=1)
			x = self.drop_after_time(x)
		
		return x, b

	def forward(self, x):
		x, b = self.prepare_tokens(x)
		# Video transformer forward
		x = self.transformer_layers(x)

		if self.attention_type == 'space_only':
			x = rearrange(x, '(b t) p d -> b t p d', b=b)
			x = reduce(x, 'b t p d -> b p d', 'mean')

		x = self.norm(x)
		# Return Class Token
		if self.return_cls_token:
			return x[:, 0]
		else:
			return x[:, 1:].mean(1)

	def get_last_selfattention(self, x):
		x, b = self.prepare_tokens(x)
		x = self.transformer_layers(x, return_attention=True)
		return x

def get_vit_base_patch16_224(**kwargs):
	vit = TimeSformer(num_frames=kwargs['num_frames'], pretrain_pth=kwargs['pretrain_pth'], weights_from=kwargs['weights_from'],
					  img_size=kwargs['img_size'], attention_type=kwargs['attention_type'], patch_size=16, embed_dims=768, num_heads=12,
					  in_channels=3, num_transformer_layers=12, conv_type='Conv2d', dropout_p=0., norm_layer=nn.LayerNorm,
					  copy_strategy='repeat', use_learnable_pos_emb=True, return_cls_token=True)
	return vit

class ViViT(nn.Module):
	"""ViViT. A PyTorch impl of `ViViT: A Video Vision Transformer`
		<https://arxiv.org/abs/2103.15691>

	Args:
		num_frames (int): Number of frames in the video.
		img_size (int | tuple): Size of input image.
		patch_size (int): Size of one patch.
		pretrained (str | None): Name of pretrained model. Default: None.
		embed_dims (int): Dimensions of embedding. Defaults to 768.
		num_heads (int): Number of parallel attention heads. Defaults to 12.
		num_transformer_layers (int): Number of transformer layers. Defaults to 12.
		in_channels (int): Channel num of input features. Defaults to 3.
		dropout_p (float): Probability of dropout layer. Defaults to 0..
		tube_size (int): Dimension of the kernel size in Conv3d. Defaults to 2.
		conv_type (str): Type of the convolution in PatchEmbed layer. Defaults to Conv3d.
		attention_type (str): Type of attentions in TransformerCoder. Choices
			are 'divided_space_time', 'fact_encoder' and 'joint_space_time'.
			Defaults to 'fact_encoder'.
		norm_layer (dict): Config for norm layers. Defaults to nn.LayerNorm.
		copy_strategy (str): Copy or Initial to zero towards the new additional layer.
		extend_strategy (str): How to initialize the weights of Conv3d from pre-trained Conv2d.
		use_learnable_pos_emb (bool): Whether to use learnable position embeddings.
		return_cls_token (bool): Whether to use cls_token to predict class label.
	"""
	supported_attention_types = [
		'fact_encoder', 'joint_space_time', 'divided_space_time'
	]

	def __init__(self,
				 num_frames,
				 img_size=224,
				 patch_size=16,
				 pretrain_pth=None,
				 weights_from='imagenet',
				 embed_dims=768,
				 num_heads=12,
				 num_transformer_layers=12,
				 in_channels=3,
				 dropout_p=0.,
				 tube_size=2,
				 conv_type='Conv3d',
				 attention_type='fact_encoder',
				 norm_layer=nn.LayerNorm,
				 copy_strategy='repeat',
				 extend_strategy='temporal_avg',
				 use_learnable_pos_emb=True,
				 return_cls_token=True,
				 **kwargs):
		super().__init__()
		assert attention_type in self.supported_attention_types, (
			f'Unsupported Attention Type {attention_type}!')
		
		num_frames = num_frames//tube_size
		self.num_frames = num_frames
		self.pretrain_pth = pretrain_pth
		self.weights_from = weights_from
		self.embed_dims = embed_dims
		self.num_transformer_layers = num_transformer_layers
		self.attention_type = attention_type
		self.conv_type = conv_type
		self.copy_strategy = copy_strategy
		self.extend_strategy = extend_strategy
		self.tube_size = tube_size
		self.num_time_transformer_layers = 0
		self.use_learnable_pos_emb = use_learnable_pos_emb
		self.return_cls_token = return_cls_token

		#tokenize & position embedding
		self.patch_embed = PatchEmbed(
			img_size=img_size,
			patch_size=patch_size,
			in_channels=in_channels,
			embed_dims=embed_dims,
			tube_size=tube_size,
			conv_type=conv_type)
		num_patches = self.patch_embed.num_patches
		
		if self.attention_type == 'divided_space_time':
			# Divided Space Time Attention - Model 3
			operator_order = ['time_attn','space_attn','ffn']
			container = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=operator_order)

			transformer_layers = container
		elif self.attention_type == 'joint_space_time':
			# Joint Space Time Attention - Model 1
			operator_order = ['self_attn','ffn']
			container = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=operator_order)
			
			transformer_layers = container
		else:
			# Divided Space Time Transformer Encoder - Model 2
			transformer_layers = nn.ModuleList([])
			self.num_time_transformer_layers = 4
			
			spatial_transformer = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=['self_attn','ffn'])
			
			temporal_transformer = TransformerContainer(
				num_transformer_layers=self.num_time_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=['self_attn','ffn'])

			transformer_layers.append(spatial_transformer)
			transformer_layers.append(temporal_transformer)
 
		self.transformer_layers = transformer_layers
		self.norm = norm_layer(embed_dims, eps=1e-6)
		
		self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dims))
		# whether to add one cls_token in temporal pos_enb
		if attention_type == 'fact_encoder':
			num_frames = num_frames + 1
			num_patches = num_patches + 1
			self.use_cls_token_temporal = False
		else:
			self.use_cls_token_temporal = operator_order[-2] == 'time_attn'
			if self.use_cls_token_temporal:
				num_frames = num_frames + 1
			else:
				num_patches = num_patches + 1

		if use_learnable_pos_emb:
			self.pos_embed = nn.Parameter(torch.zeros(1,num_patches,embed_dims))
			self.time_embed = nn.Parameter(torch.zeros(1,num_frames,embed_dims))
		else:
			self.pos_embed = get_sine_cosine_pos_emb(num_patches,embed_dims)
			self.time_embed = get_sine_cosine_pos_emb(num_frames,embed_dims)
		self.drop_after_pos = nn.Dropout(p=dropout_p)
		self.drop_after_time = nn.Dropout(p=dropout_p)

		self.init_weights()

	def init_weights(self):
		if self.use_learnable_pos_emb:
			#trunc_normal_(self.pos_embed, std=.02)
			#trunc_normal_(self.time_embed, std=.02)
			nn.init.trunc_normal_(self.pos_embed, std=.02)
			nn.init.trunc_normal_(self.time_embed, std=.02)
		trunc_normal_(self.cls_token, std=.02)
		
		if self.pretrain_pth is not None:
			if self.weights_from == 'imagenet':
				init_from_vit_pretrain_(self,
										self.pretrain_pth,
										self.conv_type,
										self.attention_type,
										self.copy_strategy,
										self.extend_strategy, 
										self.tube_size, 
										self.num_time_transformer_layers)
			elif self.weights_from == 'kinetics':
				init_from_kinetics_pretrain_(self,
											 self.pretrain_pth)
			else:
				raise TypeError(f'not support the pretrained weight {self.pretrain_pth}')
			# for p in self.parameters():
			# 	p.requires_grad = False

	@torch.jit.ignore
	def no_weight_decay_keywords(self):
		return {'pos_embed', 'cls_token', 'mask_token'}

	def prepare_tokens(self, x):
		#Tokenize
		b = x.shape[0]
		x = self.patch_embed(x)
		
		# Add Position Embedding
		cls_tokens = repeat(self.cls_token, 'b ... -> (repeat b) ...', repeat=x.shape[0])
		if self.use_cls_token_temporal:
			if self.use_learnable_pos_emb:
				x = x + self.pos_embed
			else:
				x = x + self.pos_embed.type_as(x).detach()
			x = torch.cat((cls_tokens, x), dim=1)
		else:
			x = torch.cat((cls_tokens, x), dim=1)
			if self.use_learnable_pos_emb:
				x = x + self.pos_embed
			else:
				x = x + self.pos_embed.type_as(x).detach()
		x = self.drop_after_pos(x)

		# Add Time Embedding
		if self.attention_type != 'fact_encoder':
			cls_tokens = x[:b, 0, :].unsqueeze(1)
			if self.use_cls_token_temporal:
				x = rearrange(x[:, 1:, :], '(b t) p d -> (b p) t d', b=b)
				cls_tokens = repeat(cls_tokens,
									'b ... -> (repeat b) ...',
									repeat=x.shape[0]//b)
				x = torch.cat((cls_tokens, x), dim=1)
				if self.use_learnable_pos_emb:
					x = x + self.time_embed
				else:
					x = x + self.time_embed.type_as(x).detach()
				cls_tokens = x[:b, 0, :].unsqueeze(1)
				x = rearrange(x[:, 1:, :], '(b p) t d -> b (p t) d', b=b)
				x = torch.cat((cls_tokens, x), dim=1)
			else:
				x = rearrange(x[:, 1:, :], '(b t) p d -> (b p) t d', b=b)
				if self.use_learnable_pos_emb:
					x = x + self.time_embed
				else:
					x = x + self.time_embed.type_as(x).detach()
				x = rearrange(x, '(b p) t d -> b (p t) d', b=b)
				x = torch.cat((cls_tokens, x), dim=1)
			x = self.drop_after_time(x)
		
		return x, cls_tokens, b

	def forward(self, x):
		x, cls_tokens, b = self.prepare_tokens(x)
		
		if self.attention_type != 'fact_encoder':
			x = self.transformer_layers(x)
		else:
			# fact encoder - CRNN style
			spatial_transformer, temporal_transformer, = *self.transformer_layers,
			x = spatial_transformer(x)
			
			# Add Time Embedding
			cls_tokens = x[:b, 0, :].unsqueeze(1)
			x = rearrange(x[:, 1:, :], '(b t) p d -> b t p d', b=b)
			x = reduce(x, 'b t p d -> b t d', 'mean')
			x = torch.cat((cls_tokens, x), dim=1)
			if self.use_learnable_pos_emb:
				x = x + self.time_embed
			else:
				x = x + self.time_embed.type_as(x).detach()
			x = self.drop_after_time(x)
			
			x = temporal_transformer(x)

		x = self.norm(x)
		# Return Class Token
		if self.return_cls_token:
			return x[:, 0]
		else:
			return x[:, 1:].mean(1)

	def get_last_selfattention(self, x, spatial=False):
		x, cls_tokens, b = self.prepare_tokens(x)
		
		if self.attention_type != 'fact_encoder':
			x = self.transformer_layers(x, return_attention=True)
		else:
			# fact encoder - CRNN style
			spatial_transformer, temporal_transformer, = *self.transformer_layers,
			x = spatial_transformer(x, return_attention=spatial)

			if spatial:
				return x[:,:,1:,1:]
			
			# Add Time Embedding
			cls_tokens = x[:b, 0, :].unsqueeze(1)
			x = rearrange(x[:, 1:, :], '(b t) p d -> b t p d', b=b)
			x = reduce(x, 'b t p d -> b t d', 'mean')
			x = torch.cat((cls_tokens, x), dim=1)
			if self.use_learnable_pos_emb:
				x = x + self.time_embed
			else:
				x = x + self.time_embed.type_as(x).detach()
			x = self.drop_after_time(x)
			print(x.shape)
			x = temporal_transformer(x, return_attention=True)
		return x


def parse_args():
	parser = argparse.ArgumentParser(description='lr receiver')

	# Model
	parser.add_argument(
		'-arch', type=str, default='mvit',
		help='the choosen model arch from [timesformer, vivit]')
	# Training/Optimization parameters
	parser.add_argument(
		'-optim_type', type=str, default='adamw',
		help='the optimizer using in the training')
	parser.add_argument(
		'-lr', type=float, default=0.0005,
		help='the initial learning rate')
	parser.add_argument(
		'-layer_decay', type=float, default=1,
		help='the value of layer_decay')
	parser.add_argument(
		'-weight_decay', type=float, default=0.05, 
		help="""Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.""")
	
	args = parser.parse_args()
	
	return args
