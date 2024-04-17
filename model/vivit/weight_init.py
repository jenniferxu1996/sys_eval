import math
import re
import warnings

from einops import repeat
import torch
import torch.nn as nn

from .utils import print_on_rank_zero


def show_state_dict(state_dict):
	for name, value in state_dict.items():
		print(name)


def replace_state_dict(state_dict):
	for old_key in list(state_dict.keys()):
		if old_key.startswith('model'):
			new_key = old_key[6:] # skip 'model.'
			if 'in_proj' in new_key:
				new_key = new_key.replace('in_proj_', 'qkv.') #in_proj_weight -> qkv.weight
			elif 'out_proj' in new_key:
				new_key = new_key.replace('out_proj', 'proj')
			state_dict[new_key] = state_dict.pop(old_key)
		else: # cls_head
			new_key = old_key[9:]
			state_dict[new_key] = state_dict.pop(old_key)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

    
@torch.no_grad()   
def constant_init_(tensor, constant_value=0):
	nn.init.constant_(tensor, constant_value)
	

@torch.no_grad()
def kaiming_init_(tensor,
                  a=0,
                  mode='fan_out',
                  nonlinearity='relu',
                  distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
    	nn.init.kaiming_uniform_(
    		tensor, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
    	nn.init.kaiming_normal_(
    		tensor, a=a, mode=mode, nonlinearity=nonlinearity)


@torch.no_grad()	
def init_from_vit_pretrain_(module,
						    pretrained,
						    conv_type, 
						    attention_type, 
						    copy_strategy,
						    extend_strategy='temporal_avg',
						    tube_size=2, 
						    num_time_transformer_layers=4):

	if isinstance(pretrained, str):
		if torch.cuda.is_available():
			state_dict = torch.load(pretrained)
		else:
			state_dict = torch.load(pretrained, map_location=torch.device('cpu'))

		if 'state_dict' in state_dict:
			state_dict = state_dict['state_dict']

		old_state_dict_keys = list(state_dict.keys())
		for old_key in old_state_dict_keys:
			# extend the Conv2d params to Conv3d
			if conv_type == 'Conv3d':
				if 'patch_embed.projection.weight' in old_key:
					weight = state_dict[old_key]
					new_weight = repeat(weight, 'd c h w -> d c t h w', t=tube_size)
					if extend_strategy == 'temporal_avg':
						new_weight = new_weight / tube_size
					elif extend_strategy == 'center_frame':
						new_weight.zero_()
						new_weight[:,:,tube_size//2,:,:] = weight
					state_dict[old_key] = new_weight
					continue

			# modify the key names of norm layers
			if attention_type == 'fact_encoder':
				new_key = old_key.replace('transformer_layers.layers',
										  'transformer_layers.0.layers')
			else:
				new_key = old_key

			if 'in_proj' in new_key:
				new_key = new_key.replace('in_proj_', 'qkv.') #in_proj_weight -> qkv.weight
			elif 'out_proj' in new_key:
				new_key = new_key.replace('out_proj', 'proj')

			if 'norms' in new_key:
				new_key = new_key.replace('norms.0', 'attentions.0.norm')
				new_key = new_key.replace('norms.1', 'ffns.0.norm')

			state_dict[new_key] = state_dict.pop(old_key)

		old_state_dict_keys = list(state_dict.keys())
		for old_key in old_state_dict_keys:
			# copy the parameters of space attention to time attention
			if attention_type == 'divided_space_time':
				if 'attentions.0' in old_key:
					new_key = old_key.replace('attentions.0',
											  'attentions.1')
					if copy_strategy == 'repeat':
						state_dict[new_key] = state_dict[old_key].clone()
					elif copy_strategy == 'set_zero':
						state_dict[new_key] = state_dict[old_key].clone().zero_()
			# copy the part of parameters of space attention to time attention
			elif attention_type == 'fact_encoder':
				pattern = re.compile(r'(?<=layers.)\d+')
				matchObj = pattern.findall(old_key)
				if len(matchObj) > 1 and int(matchObj[1]) < num_time_transformer_layers:
					new_key = old_key.replace('transformer_layers.0.layers',
											  'transformer_layers.1.layers')
					if copy_strategy == 'repeat':
						state_dict[new_key] = state_dict[old_key].clone()
					elif copy_strategy == 'set_zero':
						state_dict[new_key] = state_dict[old_key].clone().zero_()

		missing_keys,unexpected_keys = module.load_state_dict(state_dict, strict=False)
		#print(f'missing_keys:{missing_keys}\n unexpected_keys:{unexpected_keys}')
		print_on_rank_zero(f'missing_keys:{missing_keys}\n '
						   f'unexpected_keys:{unexpected_keys}')


@torch.no_grad()
def init_from_mae_pretrain_(module,
							pretrained,
							conv_type, 
							attention_type, 
							copy_strategy,
							extend_strategy='temporal_avg',
							tube_size=2, 
							num_time_transformer_layers=4):

	if isinstance(pretrained, str):
		if torch.cuda.is_available():
			state_dict = torch.load(pretrained)
		else:
			state_dict = torch.load(pretrained, map_location=torch.device('cpu'))
		
		if 'model' in state_dict:
			state_dict = state_dict['model']
		
		# adjust to our module
		old_state_dict_keys = list(state_dict.keys())
		for old_key in old_state_dict_keys:
			if 'decoder' in old_key:
				state_dict.pop(old_key)
				continue
			
			# extend the Conv2d params to Conv3d
			if 'encoder.patch_embed.proj' in old_key:
				new_key = old_key.replace('encoder.patch_embed.proj',
										  'patch_embed.projection')
				if conv_type == 'Conv3d' and 'weight' in old_key:
					weight = state_dict[old_key]
					new_weight = repeat(weight, 'd c h w -> d c t h w', t=tube_size)
					if extend_strategy == 'temporal_avg':
						new_weight = new_weight / tube_size
					elif extend_strategy == 'center_frame':
						new_weight.zero_()
						new_weight[:,:,tube_size//2,:,:] = weight
					state_dict.pop(old_key)
					state_dict[new_key] = new_weight
				else:
					state_dict[new_key] = state_dict.pop(old_key)
				continue

			# modify the key names of norm layers
			if attention_type == 'fact_encoder':
				new_key = old_key.replace('encoder.blocks',
										  'transformer_layers.0.layers')
			else:
				new_key = old_key.replace('encoder.blocks',
										  'transformer_layers.layers')

			if 'norm' in new_key:
				new_key = new_key.replace('norm1', 'attentions.0.norm')
				new_key = new_key.replace('norm2', 'ffns.0.norm')
			elif 'attn' in new_key:
				#new_key = new_key.replace('attn.qkv.weight',
				#						  'attentions.0.attn.in_proj_weight')
				#new_key = new_key.replace('attn.proj',
				#						  'attentions.0.attn.out_proj')
				if 'q_bias' in new_key:
					pattern = re.compile(r'(?<=blocks.)\d+')
					matchObj = pattern.findall(old_key)
					block_id = int(matchObj[0])
					q_bias = state_dict[f'encoder.blocks.{block_id}.attn.q_bias']
					v_bias = state_dict[f'encoder.blocks.{block_id}.attn.v_bias']
					weight = torch.cat((q_bias, 
										torch.zeros_like(q_bias, requires_grad=False),
										v_bias))
					new_key = new_key.replace('attn.q_bias',
											  #'attentions.0.attn.in_proj_bias')
											  'attentions.0.attn.qkv.bias')
					state_dict.pop(f'encoder.blocks.{block_id}.attn.q_bias')
					state_dict.pop(f'encoder.blocks.{block_id}.attn.v_bias')
					state_dict[new_key] = weight
					continue
				elif 'v_bias' in new_key:
					continue
			elif 'mlp' in new_key:
				new_key = new_key.replace('mlp.fc1', 'ffns.0.layers.0.0')
				new_key = new_key.replace('mlp.fc2', 'ffns.0.layers.1')
			
			if 'encoder.norm' in old_key:
				new_key = old_key.replace('encoder.norm',
										  'norm')

			state_dict[new_key] = state_dict.pop(old_key)

		# copy to new layer
		old_state_dict_keys = list(state_dict.keys())
		for old_key in old_state_dict_keys:
			# copy the parameters of space attention to time attention
			if attention_type == 'divided_space_time':
				if 'attentions.0' in old_key:
					new_key = old_key.replace('attentions.0',
											  'attentions.1')
					if copy_strategy == 'repeat':
						state_dict[new_key] = state_dict[old_key].clone()
					elif copy_strategy == 'set_zero':
						state_dict[new_key] = state_dict[old_key].clone().zero_()
			# copy the part of parameters of space attention to time attention
			elif attention_type == 'fact_encoder':
				pattern = re.compile(r'(?<=layers.)\d+')
				matchObj = pattern.findall(old_key)
				if len(matchObj) > 1 and int(matchObj[1]) < num_time_transformer_layers:
					new_key = old_key.replace('transformer_layers.0.layers', 
											  'transformer_layers.1.layers')
					if copy_strategy == 'repeat':
						state_dict[new_key] = state_dict[old_key].clone()
					elif copy_strategy == 'set_zero':
						state_dict[new_key] = state_dict[old_key].clone().zero_()

		missing_keys,unexpected_keys = module.load_state_dict(state_dict, strict=False)
		#print(f'missing_keys:{missing_keys}\n unexpected_keys:{unexpected_keys}')
		print_on_rank_zero(f'missing_keys:{missing_keys}\n '
						   f'unexpected_keys:{unexpected_keys}')
						   

def init_from_kinetics_pretrain_(module, pretrain_pth):
	if torch.cuda.is_available():
		state_dict = torch.load(pretrain_pth)
	else:
		state_dict = torch.load(pretrain_pth, map_location=torch.device('cpu'))
	if 'state_dict' in state_dict:
		state_dict = state_dict['state_dict']
	
	replace_state_dict(state_dict)
	msg = module.load_state_dict(state_dict, strict=False)
	print_on_rank_zero(msg)