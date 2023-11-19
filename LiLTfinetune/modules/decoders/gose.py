import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertAttention, BertSelfAttention




class BiaffineAttention(torch.nn.Module):
    """Implements a biaffine attention operator for binary relation classification.

    PyTorch implementation of the biaffine attention operator from "End-to-end neural relation
    extraction using deep biaffine attention" (https://arxiv.org/abs/1812.11275) which can be used
    as a classifier for binary relation classification.

    Args:
        in_features (int): The size of the feature dimension of the inputs.
        out_features (int): The size of the feature dimension of the output.

    Shape:
        - x_1: `(N, *, in_features)` where `N` is the batch dimension and `*` means any number of
          additional dimensisons.
        - x_2: `(N, *, in_features)`, where `N` is the batch dimension and `*` means any number of
          additional dimensions.
        - Output: `(N, *, out_features)`, where `N` is the batch dimension and `*` means any number
            of additional dimensions.

    Examples:
        >>> batch_size, in_features, out_features = 32, 100, 4
        >>> biaffine_attention = BiaffineAttention(in_features, out_features)
        >>> x_1 = torch.randn(batch_size, in_features)
        >>> x_2 = torch.randn(batch_size, in_features)
        >>> output = biaffine_attention(x_1, x_2)
        >>> print(output.size())
        torch.Size([32, 4])
    """

    def __init__(self, in_features, out_features):
        super(BiaffineAttention, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.bilinear = torch.nn.Bilinear(in_features, in_features, out_features, bias=False)
        self.linear = torch.nn.Linear(2 * in_features, out_features, bias=True)

        self.reset_parameters()

    def forward(self, x_1, x_2):
        return self.bilinear(x_1, x_2) + self.linear(torch.cat((x_1, x_2), dim=-1))

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()

from torch.nn.modules.batchnorm import _BatchNorm
from timm.models.layers import DropPath, trunc_normal_, drop_path

def get_relative_position_index(
        win_h: int,
        win_w: int
) -> torch.Tensor:
    """ Function to generate pair-wise relative position index for each token inside the window.
        Taken from Timms Swin V1 implementation.
    Args:
        win_h (int): Window/Grid height.
        win_w (int): Window/Grid width.
    Returns:
        relative_coords (torch.Tensor): Pair-wise relative position indexes [height * width, height * width].
    """
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)

def window_reverse(
        windows: torch.Tensor,
        original_size,
        window_size=(7, 7)
) -> torch.Tensor:
    """ Reverses the window partition.
    Args:
        windows (torch.Tensor): Window tensor of the shape [B * windows, window_size[0] * window_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        window_size (Tuple[int, int], optional): Window size which have been applied. Default (7, 7)
    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, original_size[0] * original_size[1], C].
    """
    # Get height and width
    H, W = original_size
    # Compute original batch size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # Fold grid tensor
    output = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 1, 3, 2, 4, 5).reshape(B, H * W, -1)
    return output
    

class Attention(nn.Module):
    def __init__(self, dim, num_tokens=1, num_heads=8, window_size=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads # S
        head_dim = dim // num_heads 
        self.num_tokens = num_tokens
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = qk_scale or head_dim ** -0.5
        self.kv_prefix = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        # positional embedding
        # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.cnt = 0

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)


    def forward_local(self, q, k, v, H, W, mask, past_key_value=None):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """

        B, num_heads, N, C = q.shape
        ws = self.window_size
        h_group, w_group = H // ws, W // ws

        mask = mask.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        mask = mask.view(-1, num_heads, ws*ws, C)
        q = q.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        q = q.view(-1, num_heads, ws*ws, C)
        k = k.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        k = k.view(-1, num_heads, ws*ws, C)
        v = v.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        v = v.view(-1, num_heads, ws*ws, C)
        if past_key_value is not None:
            k_prefix, v_prefix = past_key_value
            k_prefix = k_prefix.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
            k_prefix = k_prefix.view(-1, num_heads, ws*ws, C)
            v_prefix = v_prefix.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
            v_prefix = v_prefix.view(-1, num_heads, ws*ws, C)
            k = torch.cat([k_prefix, k], dim=-2)
            v = torch.cat([v_prefix, v], dim=-2)
            mask_prefix = torch.cat([mask, mask], dim=-2)
            attn_mask = (mask @ mask_prefix.transpose(-2, -1)) / C
        else:
            attn_mask = (mask @ mask.transpose(-2, -1)) / C
            
        attn_mask = attn_mask.bool() == False
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill_(attn_mask, -10000)
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws*ws, -1)

        # reverse
        x = window_reverse(x, (H, W), (ws, ws))
        return x

    def forward_global_aggregation(self, q, k, v, mask, global_mask=None,key_num=None, value_num=None):
        """
        q: global tokens
        k: image tokens
        v: image tokens
        """
        B, _, N, C = q.shape
        q_shape = torch.ones(q.shape,device=q.device)
        attn_mask = (q_shape @ mask.transpose(-2, -1)) / C
        if global_mask is not None:
            attn_mask = global_mask * attn_mask
        attn_mask = attn_mask.bool() == False
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill_(attn_mask,-10000)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x
    def forward_global_broadcast(self, q, k, v, mask):
        """
        q: image tokens
        k: global tokens
        v: global tokens
        """
        B, num_heads, N, C = q.shape
        k_shape = torch.ones(k.shape,device=k.device)
        attn_mask = (mask @ k_shape.transpose(-2, -1)) / C
        attn_mask = attn_mask.bool() == False
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill_(attn_mask,-10000)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    def forward(self, x, H, W, mask, global_mask, layout_prefix=None, key_num=None, value_num=None):
        B, N, C = x.shape
        NC = self.num_tokens * self.num_tokens

        x_img, x_global = x[:, NC:], x[:, :NC]
        x_img = x_img.view(B, H, W, C)
        pad_l = pad_t = 0
        ws = self.window_size#7
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        x_img = F.pad(x_img, (0, 0, pad_l, pad_r, pad_t, pad_b))
        Hp, Wp = x_img.shape[1], x_img.shape[2]
        x_img = x_img.view(B, -1, C)
        x = torch.cat([x_global, x_img], dim=1)

        qkv = self.qkv(x)                                                                                   # q    B,M*M,C -> B,M*M,(q,k,v),n_heads,C//n_heads
        q, k, v = qkv.view(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)  # q               B,n_heads,M*M,c//n_heads
        mask = mask.view(B, -1, self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)                   # mask B,M,M,C -> B,n_heads,M*M,C//n_heads
        if global_mask is not None:
            global_mask = global_mask.view(B, self.num_heads, NC, -1) 
 
        q_img, k_img, v_img = q[:, :, NC:], k[:, :, NC:], v[:, :, NC:]
        q_cls, _, _ = q[:, :, :NC], k[:, :, :NC], v[:, :, :NC]
        if layout_prefix is not None:

            kv_prefix = self.kv_prefix(layout_prefix)
            k_prefix, v_prefix = kv_prefix.view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
            past_key_value = [k_prefix, v_prefix]
        else:
            past_key_value = None
        

        x_img = self.forward_local(q_img, k_img, v_img, Hp, Wp, mask, past_key_value=past_key_value)

        x_img = x_img.view(B, Hp, Wp, -1)[:, :H, :W].reshape(B, H*W, -1)
        q_img = q_img.reshape(B, self.num_heads, Hp, Wp, -1)[:, :, :H, :W].reshape(B, self.num_heads, H*W, -1)
        k_img = k_img.reshape(B, self.num_heads, Hp, Wp, -1)[:, :, :H, :W].reshape(B, self.num_heads, H*W, -1)
        v_img = v_img.reshape(B, self.num_heads, Hp, Wp, -1)[:, :, :H, :W].reshape(B, self.num_heads, H*W, -1)

        x_cls = self.forward_global_aggregation(q_cls, k_img, v_img, mask, global_mask,key_num, value_num)
        k_cls, v_cls = self.kv_global(x_cls).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)

        x_img = x_img + self.forward_global_broadcast(q_img, k_cls, v_cls, mask)
        x = torch.cat([x_cls, x_img], dim=1)
        x = self.proj(x)
        return x

class GRU(torch.nn.Module):
    def __init__(self, dim):
        super(GRU, self).__init__()

        self.dim = dim

        self.W_xhr = torch.nn.Linear(dim*2,dim)
        self.sigmoid = torch.nn.Sigmoid()

     
        
    def forward(self, x, x_new):
   
        xh = torch.cat([x,x_new],dim=-1)
        Rt = self.sigmoid(self.W_xhr(xh)) 
        H = x + Rt * x_new 
        return H    
 
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import copy
class GOSE(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.rounds = args.rounds+1
        self.norm = False
        self.hidden_size = 960
        self.hidden_dropout_prob = 0.5
 
        self.pooling_mode = args.pooling_mode
        self.use_attn = args.use_attn
        self.loss_fct = CrossEntropyLoss()
        self.use_prefix = args.use_prefix
        self.use_global_mask = args.use_global_mask
        self.use_gate = args.use_gate
        print(f"**********************************Use_Attn: {self.use_attn}************************************")
        print(f"**********************************Use_Prefix: {self.use_prefix}********************************")
        print(f"**********************************Use_Global_Mask: {self.use_global_mask}**********************")
        print(f"**********************************Use_Gate: {self.use_gate}************************************")
        print(f"**********************************Pooling_Mode: {self.pooling_mode}****************************")
        print(f"**********************************Iterative_Rounds: {self.rounds-1}****************************")

        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.elu=nn.ELU()
        self.biaffine = BiaffineAttention(self.hidden_size//2 , 2)

        self.ffn = nn.Linear(2, self.hidden_size//2)
        self.ffn_key = nn.Linear(self.hidden_size, self.hidden_size//2)
        self.ffn_value = nn.Linear(self.hidden_size, self.hidden_size//2)

        self.dim = self.hidden_size //2
        self.num_heads = 1
        self.num_tokens = 8 
        self.window_size = args.window_size 
        self.qkv_bias = False
        self.drop = 0
        self.attn_drop = 0
        self.drop_path = 0
        self.max_len = args.window_size ** 2
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.global_token_num = self.num_tokens**2
        self.global_token = nn.Parameter(torch.zeros(1, self.global_token_num, self.hidden_size //2))
        self.attn = Attention(self.dim,num_heads=self.num_heads, num_tokens=self.num_tokens, 
                              window_size=self.window_size,qkv_bias=self.qkv_bias, 
                              attn_drop=self.attn_drop, proj_drop=self.drop)

        self.cnt = 0
        self.loss_fcn = CrossEntropyLoss()
        self.normal = True
        self.dummy_vec = nn.Parameter(torch.Tensor(1, self.hidden_size//2))
        nn.init.normal_(self.dummy_vec)

        self.gru = GRU(self.hidden_size//2) 
    
        self.axis_dis_fn = nn.Linear(1, self.hidden_size//12)
        self.axis_angle_fn = nn.Linear(1, self.hidden_size//12)

        self.global_mask = self.create_global_mask()
    
    def create_global_mask(self):
        global_mask = torch.zeros(self.global_token_num, self.max_len, self.max_len).cuda()
        step = self.num_tokens
        for idx in range(self.global_token_num):
            row_ids = idx // self.num_tokens
            column_ids = idx % self.num_tokens
            row_start = row_ids * step
            column_start = column_ids * step
            global_mask[idx, row_start:row_start+self.num_tokens,:] = 1
            global_mask[idx, :, column_start:column_start+self.num_tokens] = 1
        return global_mask
        
    def get_entities_kv_index_list(self, entities):

        M = self.max_len
        entities_label = entities['label']

        entities_key_index = [index for index,label in enumerate(entities_label) if label == 1 ]
        entities_value_index = [index for index,label in enumerate(entities_label) if label == 2 ] 
        key_num, value_num =  len(entities_key_index),len(entities_value_index)
        '''
        in re.py
                if len(all_possible_relations) == 0:
                all_possible_relations = set([(0, 1)])
        '''
        if key_num * value_num == 0:
            entities_key_index = [0]
            entities_value_index = [1]
        if key_num > M :
            entities_key_index = entities_key_index[:M]
            self.normal = False
        if value_num > M :
            entities_value_index = entities_value_index[:M]
            self.normal = False

        return entities_key_index, entities_value_index

    
    def forward(self, hidden_state, entities,relations, bbox):
        self.cnt += 1
        B ,_ ,H = hidden_state.shape
        M = self.max_len
        device = hidden_state.device

        loss = 0
        all_pred_relations = []

        key_repr_list = []
        value_repr_list = []
        key_mask_list = []
        value_mask_list = []
        key_bbox_list, value_bbox_list = [], []
        for b in range(B):

            if len(entities[b]["start"]) <= 2:
                entities[b] = {"end": [1, 1], "label": [0, 0], "start": [0, 0]}
            
            entities_key_index, entities_value_index = self.get_entities_kv_index_list(entities[b])
            entities_first_token_index = torch.tensor(entities[b]['start'])
            
            entities_key_first_token_index = entities_first_token_index[entities_key_index]
            entities_value_first_token_index = entities_first_token_index[entities_value_index]

            key_repr = hidden_state[b][entities_key_first_token_index,:]
            value_repr = hidden_state[b][entities_value_first_token_index,:]
            
            key_num,value_num =  key_repr.shape[0],value_repr.shape[0]

            
            key_mask_list.append(torch.tensor([[1.]] * key_num + [[0.]] * (M - key_num),device=device).repeat(1,H//2))
            value_mask_list.append(torch.tensor([[1.]] * value_num + [[0.]] * (M - value_num),device=device).repeat(1,H//2))
          
            key_repr_list.append(F.pad(key_repr,(0, 0, 0, M - key_num)))
            value_repr_list.append(F.pad(value_repr,(0, 0, 0, M - value_num)))
 
            key_bbox = bbox[b][entities_key_first_token_index]
            value_bbox = bbox[b][entities_value_first_token_index]
            key_bbox_list.append(F.pad(key_bbox,(0, 0, 0, M - key_num)))
            value_bbox_list.append(F.pad(value_bbox,(0, 0, 0, M - value_num)))

        key_repr = torch.stack(key_repr_list,dim=0) 
        key_mask = torch.stack(key_mask_list,dim=0)
       
        value_repr = torch.stack(value_repr_list,dim=0)
        value_mask = torch.stack(value_mask_list,dim=0)
        

        table_mask = key_mask.unsqueeze(2).repeat(1,1,M,1)\
                    *value_mask.unsqueeze(1).repeat(1,M,1,1)

        if self.use_global_mask:
            global_mask = self.global_mask.unsqueeze(0).repeat(B,1,1,1) 
        else:
            global_mask = None
                    
    
        key_mask = key_mask[:,:,0].bool()
        value_mask = value_mask[:,:,0].bool()

        key_ffn = self.ffn_key(key_repr)
        value_ffn = self.ffn_value(value_repr)
        
        if self.norm == True:
            key_ffn = self.norm1(key_repr)
            value_ffn = self.norm1(value_repr)
  
        global_token = self.global_token.expand(B, -1, -1)

        key_bbox = torch.stack(key_bbox_list, dim=0) 
        value_bbox = torch.stack(value_bbox_list, dim=0)   
        layout_repr = self.calc_layout(key_bbox, value_bbox)
        layout_repr = layout_repr * table_mask
        layout_repr = layout_repr.view(B,M*M,H//2)
        for i in range(self.rounds):
            '''
            method 1 with biaffine 
            
            table_mask.shape    B M M H/2  -> B M M H    (M=64)
            table_logits.shape  B M M H/2  -> B M M 2
                                B M M 2    -> B M M H
            attention input     B  (64+1)*64  384
                table input     64 * 64 
                window_size     8
                  token_num     64/8 * 64/8 = 64
            '''
            
            table_logits = self.biaffine(key_ffn.unsqueeze(2).repeat(1,1,M,1),
                                         value_ffn.unsqueeze(1).repeat(1,M,1,1))
            if i < self.rounds-1 :
                table_logits = self.ffn(table_logits) * table_mask
                
                if self.use_attn:
                    table_logits = table_logits.view(B,M*M,H//2)
                    
         
                    table_logits = torch.cat((global_token, table_logits), dim=1)
       
                    if self.use_prefix:
                        table_logits =  self.attn(table_logits, M, M, table_mask, global_mask, layout_prefix=layout_repr, key_num=key_num, value_num=value_num)
                    else:
                        table_logits =  self.attn(table_logits, M, M, table_mask, global_mask, layout_prefix=None)
                    global_token_new = table_logits[:,:self.num_tokens*self.num_tokens,:]
                    table_logits = table_logits[:,self.num_tokens*self.num_tokens:,:]
                    table_logits = table_logits.view(B,M,M,H//2)
                    table_logits = table_logits * table_mask
                    global_token = global_token + global_token_new
      
                key_new, value_new = self.get_new_repr(table_logits, key_mask, value_mask)
                if self.norm == True:
                    key_new = self.norm2(key_new)
                    value_new = self.norm2(value_new)
                if self.use_gate:
                    key_ffn = self.gru(key_ffn,key_new)
                    value_ffn = self.gru(value_ffn,value_new)
                else:
                    key_ffn = key_ffn + key_new
                    value_ffn = value_ffn + value_new
            else:
                table_logits = table_logits * table_mask[:,:,:,:2]


        loss = self.get_loss(table_logits,entities,relations,key_mask,value_mask)
        all_pred_relations = self.get_predicted_relations(table_logits,entities,key_mask,value_mask, bbox)
        return loss,all_pred_relations
    
    def calc_layout(self, head_bbox, tail_bbox):
        bsz, num, _ = head_bbox.shape
        head_bbox = head_bbox.unsqueeze(2).repeat(1,1,num,1)
        tail_bbox = tail_bbox.unsqueeze(1).repeat(1,num,1,1)
        

        head_bbox_center = torch.div(torch.cat(((head_bbox[:,:,:,0]+head_bbox[:,:,:,2]).view(-1,1), (head_bbox[:,:,:,1]+head_bbox[:,:,:,3]).view(-1,1)),dim=1), 2)
        tail_bbox_center = torch.div(torch.cat(((tail_bbox[:,:,:,0]+tail_bbox[:,:,:,2]).view(-1,1), (tail_bbox[:,:,:,1]+tail_bbox[:,:,:,3]).view(-1,1)),dim=1), 2)
        head_tail_center_dis, hea_tail_center_angle = self.axis_features(head_bbox_center, tail_bbox_center)
        head_tail_center_dis_feature = self.axis_dis_fn(head_tail_center_dis)
        head_tail_center_angle_feature = self.axis_angle_fn(hea_tail_center_angle)
     
        head_bbox_left_top = torch.cat((head_bbox[:,:,:, 0].view(-1,1), head_bbox[:,:,:, 1].view(-1,1)), dim=1)
        tail_bbox_left_top = torch.cat((tail_bbox[:,:,:, 0].view(-1,1), tail_bbox[:,:,:, 1].view(-1,1)), dim=1)
        head_tail_lt_dis, hea_tail_lt_angle = self.axis_features(head_bbox_left_top, tail_bbox_left_top)
        head_tail_lt_dis_feature = self.axis_dis_fn(head_tail_lt_dis)
        hea_tail_lt_angle_feature = self.axis_angle_fn(hea_tail_lt_angle)

        head_bbox_right_down = torch.cat((head_bbox[:,:,:, 2].view(-1,1), head_bbox[:,:,:, 3].view(-1,1)), dim=1)
        tail_bbox_right_down = torch.cat((tail_bbox[:,:,:, 2].view(-1,1), tail_bbox[:,:,:, 3].view(-1,1)), dim=1)
        head_tail_rd_dis, hea_tail_rd_angle = self.axis_features(head_bbox_right_down, tail_bbox_right_down)
        head_tail_rd_dis_feature = self.axis_dis_fn(head_tail_rd_dis)
        hea_tail_rd_angle_feature = self.axis_angle_fn(hea_tail_rd_angle)
        layout_repr = torch.cat(
                (head_tail_center_dis_feature, head_tail_center_angle_feature
                 , head_tail_lt_dis_feature, hea_tail_lt_angle_feature
                 , head_tail_rd_dis_feature, hea_tail_rd_angle_feature
                 ),
                 dim=-1
        )
        layout_repr = layout_repr.view(bsz, num, num, -1) 
        return layout_repr
        
        
    
    def axis_features(self, tmp_bbox_1, tmp_bbox_2):
        tmp_bbox_distance = torch.pow(torch.sum(torch.pow(tmp_bbox_1 - tmp_bbox_2, 2), dim=1), 0.5) #欧氏距离
        tmp_bbox_distance = tmp_bbox_distance.view(-1, 1)

        head_tail_x = tmp_bbox_1[:, 0] - tmp_bbox_2[:, 0]
        head_tail_y = tmp_bbox_1[:, 1] - tmp_bbox_2[:, 1]
        tmp_bbox_angle = torch.div(torch.atan2(head_tail_y, head_tail_x), 3.1416) #正切的角度
        tmp_bbox_angle = tmp_bbox_angle.view(-1, 1)
        return torch.div(tmp_bbox_distance, 1000), tmp_bbox_angle

    
    
    
    def get_new_repr(self, table_logits, key_mask, value_mask):
        key_repr_list = []
        value_repr_list = []
        bs,_,_,_ = table_logits.shape
        for b in range(bs):
            logit = table_logits[b][key_mask[b]]
            logit = logit[:,value_mask[b]]
            key_num, value_num, _ = logit.shape
            if self.pooling_mode == 'max':
                key_repr = logit.max(dim=1).values 
                value_repr = logit.max(dim=0).values 
            else:
                key_repr = logit.mean(dim=1)
                value_repr = logit.mean(dim=0)
            key_repr_list.append(F.pad(key_repr,(0, 0, 0, self.max_len - key_num)))
            value_repr_list.append(F.pad(value_repr,(0, 0, 0, self.max_len - value_num)))
        key_new = torch.stack(key_repr_list,dim=0) 
        value_new = torch.stack(value_repr_list,dim=0)
        return key_new, value_new
            
    def get_predicted_relations(self, logists,entities,key_mask,value_mask,bbox):
        all_pred_relations = []

        B,N,M,_=logists.shape
        for b in range(B):

            pred_relations = []
            logist = logists[b][key_mask[b]]
            logist = logist[:,value_mask[b]]
            N,M,_ = logist.shape
            
            entities_key_index, entities_value_index = self.get_entities_kv_index_list(entities[b])

            for index in range(M*N):
                key = index // M
                value = index % M
                pred_label = logist[key][value].argmax(-1)

                if pred_label == 0:
                    continue
                
                rel = {}
                rel["head_id"] = entities_key_index[key]
                rel["head"] = (entities[b]["start"][rel["head_id"]], entities[b]["end"][rel["head_id"]])
                rel["head_type"] = entities[b]["label"][rel["head_id"]]

                rel["tail_id"] = entities_value_index[value]
                rel["tail"] = (entities[b]["start"][rel["tail_id"]], entities[b]["end"][rel["tail_id"]])
                rel["tail_type"] = entities[b]["label"][rel["tail_id"]]
                rel["type"] = 1
               
                key_bbox_left_top = bbox[b][entities[b]["start"][rel["head_id"]]].tolist()[:2]
                value_bbox_left_top = bbox[b][entities[b]["start"][rel["tail_id"]]].tolist()[:2]
                rel["link"] = (tuple(key_bbox_left_top), tuple(value_bbox_left_top))
         
                pred_relations.append(rel)
            all_pred_relations.append(pred_relations)
           
        return  all_pred_relations
      
    
    def get_loss(self,logists,entities,relations,key_mask,value_mask):
        device = logists.device
        loss = 0
        B = key_mask.shape[0]
        all_logits = []
        all_labels = []
        for b in range(B):
            logist = logists[b][key_mask[b]]
            logist = logist[:,value_mask[b]]
            N,M,_ = logist.shape


            entities_key_index, entities_value_index = self.get_entities_kv_index_list(entities[b])

            entities_key_list = relations[b]['head']
            entities_value_list = relations[b]['tail']

            labels = torch.zeros(N*M).to(device).view(N,M)
            
            for i in range(len(entities_key_list)):
                try:
                    key = entities_key_index.index(entities_key_list[i])
                    value = entities_value_index.index(entities_value_list[i])
                    labels[key][value] = 1
                except:
                    continue
            
            
            labels = labels.view(-1).to(dtype=torch.long)
            logist = logist.view(N*M,-1).to(dtype=torch.float)
            all_logits.append(logist)
            all_labels.append(labels)

        all_logits = torch.cat(all_logits, 0)
        all_labels = torch.cat(all_labels, 0)
        loss = self.loss_fcn(all_logits+1e-10, all_labels)

        if (torch.isnan(loss).sum().item() > 0):
            from IPython import embed;embed();exit()
      
        return loss