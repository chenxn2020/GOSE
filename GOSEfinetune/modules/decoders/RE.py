import copy

import torch
from torch import nn
from torch.nn import CrossEntropyLoss


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



class Attention_kv(nn.Module):
    def __init__(self, dim,max_len):
        super().__init__()
        #self.num_heads = num_heads
        self.dim = dim 
        self.scale = dim ** -0.5
        self.max_len = max_len
     
        self.self_qkv = nn.Linear(dim,dim*3)
        self.cross_q =  nn.Linear(dim,dim)
        self.cross_kv = nn.Linear(dim,dim*2)

        self.ffn = nn.Linear(dim,dim) 
    def self_attention(self,x,mask):
        B, M, C = x.shape
        qkv = self.self_qkv(x)
        q,k,v = qkv.view(B,M,3,C).permute(2,0,1,3).unbind(0)
        
        attn_mask = mask @ mask.transpose(-2,-1) / C
        attn_mask = attn_mask.bool() == False 

        attn = (q @ k.transpose(-2,-1)) * self.scale 
        attn = attn.masked_fill_(attn_mask,-10000.0)
        attn = attn.softmax(dim=-1)

        x = attn @ v
        return x 
    
    def cross_attention(self,cross_x,x,mask):
        B, M, C = x.shape
        q = self.cross_q(cross_x)
        kv = self.cross_kv(x)
        k,v = kv.view(B,M,2,C).permute(2,0,1,3).unbind(0)

        attn_mask = mask @ mask.transpose(-2,-1) / C
        attn_mask = attn_mask.bool() == False 

        attn = (q @ k.transpose(-2,-1)) * self.scale 
        attn = attn.masked_fill_(attn_mask,-10000.0)
        attn = attn.softmax(dim=-1)

        x = attn @ v
        return x 
    
    def forward(self,layout_x,text_x,mask):
        mask = mask.unsqueeze(2).repeat(1,1,self.dim)
        layout_x = self.self_attention(layout_x,mask)
        merge_x = self.cross_attention(layout_x,text_x,mask)
        merge_x = self.ffn(merge_x)
        
        return merge_x

class Attention_logits(nn.Module):
    def __init__(self, dim,max_len):
        super().__init__()
        #self.num_heads = num_heads
        self.dim = dim 
        self.scale = dim ** -0.5
        self.max_len = max_len
     
        self.self_qkv = nn.Linear(dim,dim*3)
        self.cross_q =  nn.Linear(dim,dim)
        self.cross_kv = nn.Linear(dim,dim*2)

        self.ffn = nn.Linear(dim,dim) 

    def self_attention(self,x,mask):
        B, N, M, C = x.shape
        qkv = self.self_qkv(x)
        q,k,v = qkv.view(B,N,M,3,C).permute(3,0,1,2,4).unbind(0)
        
        mask = mask.unsqueeze(3).repeat(1,1,1,C)
        attn_mask = mask @ mask.transpose(-2,-1) / C
        attn_mask = attn_mask.bool() == False 

        attn = (q @ k.transpose(-2,-1)) * self.scale 
        attn = attn.masked_fill_(attn_mask,-10000.0)
        attn = attn.softmax(dim=-1)

        x = attn @ v
        return x 
    
    def forward(self,x,mask):
        #logits B N M H
        x = self.self_attention(x,mask)
        x = self.self_attention(x.transpose(-3,-2),mask.transpose(-2,-1))
        x = x.transpose(-3,-2)
        return x


from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import copy
class RE(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.cnt=0
        self.rounds = 5
        self.hidden_size = 960
        self.dim = self.hidden_size // 2
        self.hidden_dropout_prob = 0.5
        self.max_key = 64
        self.max_value = 64
        self.pooling_mode = 'max'
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fct = CrossEntropyLoss()
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.output = nn.Linear(self.dim,2)
        self.k_up = nn.Linear(2,self.dim)
        self.v_up = nn.Linear(2,self.dim)

        self.type_token = nn.Parameter(torch.normal(0,0.0002,size=(1,self.hidden_size)))
        self.biaffine_type = BiaffineAttention(self.dim , 3)
        self.biaffine =  BiaffineAttention(self.dim , 2)
        self.ffn = nn.Linear(2,self.dim)
        self.ffn_type = nn.Linear(3,self.dim)
        self.attn_type = Attention_logits(self.dim,max_len=self.max_key)
        self.attn = Attention_logits(self.dim,max_len=self.max_key)

        self.key_type_ffn = nn.Linear(self.hidden_size,self.dim )
        self.value_type_ffn = nn.Linear(self.hidden_size,self.dim )
        self.key_multi_ffn = nn.Linear(self.hidden_size,self.dim )
        self.value_multi_ffn = nn.Linear(self.hidden_size,self.dim )
        self.key_single_ffn = nn.Linear(self.hidden_size,self.dim )
        self.value_single_ffn = nn.Linear(self.hidden_size,self.dim )

        self.classifier = nn.Linear(self.dim * 2,2)
        """
        self.text_biaffine = BiaffineAttention(self.hidden_size//2 , 2)
        """
    def devide_entities(self,entities):
        """
            devide entities into keys and values according there entities label
            return entities index
        """
        entities_label_list = entities['label']
        key_index = [index for index,label in enumerate(entities_label_list) if label == 1]
        value_index =  [index for index,label in enumerate(entities_label_list) if label == 2]

        key_num = len(key_index)
        value_num = len(value_index)

        M = self.max_key
        N = self.max_value 

        if not key_num * value_num :
            key_index = [0]
            value_index = [1]
            
        if key_num > M :
            key_index = key_index[:M]
        if value_num > N:
            value_index = value_index[:N]

        return key_index, value_index 

    def padding(self,data,N):
        # padding data 2,n,768 -> 2,N,768
        n = data.shape[0]   
        dim = data.shape[1]
        device = data.device
        data = F.pad(data,(0,0,0,N-n))
        mask = torch.tensor([1.0]*n + [0.0]*(N-n),device=device)
        return data,mask 

    def type_classifier(self,key,value,key_mask,value_mask):
        key = self.key_type_ffn(key)
        value = self.value_type_ffn(value)
        
        M = self.max_key
        N = self.max_value + 1
        logits_mask = key_mask.unsqueeze(2).repeat(1,1,N) * \
                        value_mask.unsqueeze(1).repeat(1,M,1) 
        for i in range(self.rounds):
        
            logits = self.biaffine_type(key.unsqueeze(2).repeat(1,1,N,1),
                                            value.unsqueeze(1).repeat(1,M,1,1))
            if i < self.rounds-1:
                # B K V H
                logits = self.ffn_type(logits)
                logits = self.attn_type(logits,logits_mask)
                det_key,det_value = self.pooling(logits,key_mask,value_mask)
                key += det_key
                value += det_value 
            else: 
                logits = logits * logits_mask.unsqueeze(3).repeat(1,1,1,3)
        return logits 
    
    def multi_classifier(self,key,value,key_mask,value_mask):
        key = self.key_multi_ffn(key)
        value = self.value_multi_ffn(value)

        M = key.shape[1]
        N = value.shape[1]
        
        key = key.unsqueeze(2).repeat(1,1,N,1)
        value = value.unsqueeze(1).repeat(1,M,1,1)

        multi_logits = self.classifier(torch.cat([key,value],dim=-1))

        return multi_logits 
    
    def single_classifier(self,key,value,key_mask,value_mask):
        key = self.key_single_ffn(key)
        value = self.value_single_ffn(value)
        
        M = key.shape[1]
        N = value.shape[1]
     
        logits_mask = key_mask.unsqueeze(2).repeat(1,1,N) * \
                         value_mask.unsqueeze(1).repeat(1,M,1) 
        
        for i in range(self.rounds):
            logits = self.biaffine(key.unsqueeze(2).repeat(1,1,N,1),
                                            value.unsqueeze(1).repeat(1,M,1,1))
            if i < self.rounds-1:
                # B K V H
                logits = self.ffn(logits)
                logits = self.attn(logits,logits_mask)
                det_key,det_value = self.pooling(logits,key_mask,value_mask)
                key += det_key
                value += det_value 
            else: 
                logits = logits * logits_mask.unsqueeze(3).repeat(1,1,1,2) 

        return logits 
    
    def forward(self, hidden_state, entities, relations, bbox):
        self.cnt+=1
        #layout_emb,text_emb = hidden_state 
        B, max_len, H = hidden_state.shape
        device = hidden_state.device
        M = self.max_key
        N = self.max_value
        loss = 0
        all_pred_relations = []

        batch = []
        for b in range(B):
            if len(entities[b]['start']) <= 2:
                entities[b] = {"end":[1,1],"label":[0,0],"start":[0,0]}
        
            key_index,value_index = self.devide_entities(entities[b])
            start_token_index = torch.tensor(entities[b]['start'])
            key_start_token = start_token_index[key_index]
            value_start_token = start_token_index[value_index]
            #b,2,len,dim
            key = hidden_state[b][key_start_token,:]
            value =  hidden_state[b][value_start_token,:]

            key,key_mask = self.padding(key,self.max_key)
            value = torch.cat([self.type_token,value],dim=0)
            value,value_mask = self.padding(value,self.max_value+1)

            batch.append((key,value,key_mask,value_mask))
        
        
        org_key = torch.stack([item[0] for item in batch],dim=0)
        org_value = torch.stack([item[1] for item in batch],dim=0)
        key_mask = torch.stack([item[2] for item in batch],dim=0)
        value_mask = torch.stack([item[3] for item in batch],dim=0)

        type_logits = self.type_classifier(org_key,org_value,key_mask,value_mask)
        """
        self.type_token 0 - no link 
                        1 - single link 
                        2 - multilink
        B M N+1 3/
        """
       
        org_value = org_value[:,1:,:]
        value_mask = value_mask[:,1:]

        type_token = self.softmax(type_logits[:,:,0])
        key_type = type_token.argmax(dim=-1)
        #so far we can get key label to route for downstream processing
        type_drop = key_type == 0
        type_single = key_type == 1
        type_multi = key_type == 2

        #multi_key = org_key[type_multi]
        multi_logits = self.multi_classifier(org_key,org_value,key_mask,value_mask)

        key_mask = key_mask.bool() & type_single
        single_logits = self.single_classifier(org_key,org_value,key_mask,value_mask)

        type_loss = self.get_type_loss(type_logits,key_mask,entities,relations)
        multi_loss = self.get_multi_loss(multi_logits,entities,relations)
        single_loss = self.get_single_loss(single_logits,entities,relations)

        loss = type_loss + multi_loss + single_loss 
        all_pred_relations  = self.get_predicted_relations(logits,entities,relations,key_mask,value_mask)

        return loss,all_pred_relations


    def pooling(self, table_logits, key_mask, value_mask):
        key_repr_list = []
        value_repr_list = []
        bs,_,_,_ = table_logits.shape
        key_mask = key_mask.to(torch.bool)
        value_mask = value_mask.to(torch.bool)
        M = key_mask.shape[1]
        N = value_mask.shape[1]
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
            key_repr_list.append(F.pad(key_repr,(0, 0, 0, M - key_num)))
            value_repr_list.append(F.pad(value_repr,(0, 0, 0, N - value_num)))
        key_new = torch.stack(key_repr_list,dim=0) 
        value_new = torch.stack(value_repr_list,dim=0)
        return key_new, value_new
    
    def get_type_loss(self,type_logits,key_mask,entities,relations):
        # logits 2,64,65,3
        logits = self.softmax(type_logits[:,:,0])
        B = logits.shape[0]
        device = logits.device
        key_mask = key_mask.bool()
        loss_fcn = CrossEntropyLoss()

        for b in range(B):
            logit = logits[b][key_mask[b]]

            from IPython import embed;embed()
        relations 

    def get_loss(self,logits,entities,relations,key_mask,value_mask):
        loss_fcn = CrossEntropyLoss()
        B = logits.shape[0]
        device = logits.device
        loss = 0
        all_logits = []
        all_labels = []
        key_mask = key_mask.to(torch.bool)
        value_mask = value_mask.to(torch.bool)
        for b in range(B):
            logit = logits[b][key_mask[b]]
            logit = logit[:,value_mask[b]]
            M,N,_ = logit.shape 

            key_index,value_index = self.devide_entities(entities[b])
            key_list = relations[b]['head']
            value_list = relations[b]['tail']
            labels = torch.zeros(M*N,device=device).view(M,N)

            true_label = []
            for i in range(len(key_list)):
                try:
                    key = key_index.index(key_list[i])
                    value = value_index.index(value_list[i])
                    labels[key][value] = 1
                except:
                    continue
            
            labels = labels.view(-1).to(dtype=torch.long)
            logit = logit.view(M*N,-1).to(dtype=torch.float)
          

            all_logits.append(logit)
            all_labels.append(labels)

        all_logits = torch.cat(all_logits,0)
        all_labels = torch.cat(all_labels,0)

        loss = loss_fcn(all_logits+1e-10,all_labels)
        return loss 
    
    def get_predicted_relations(self,logits,entities,relations,key_mask,value_mask):
        #from IPython import embed;embed()
        softmax = nn.Softmax(dim=-1)
        all_pred_relations = []
      
        device = logits.device
        B = logits.shape[0]
      

        key_mask = key_mask.to(torch.bool)
        value_mask = value_mask.to(torch.bool)
        for b in range(B):
            pred_relations = []
            logit = logits[b][key_mask[b]]
            logit = logit[:,value_mask[b]]
    
            M,N,_ = logit.shape
        
            key_index,value_index = self.devide_entities(entities[b])
            for index in range(M*N):
                key = index // N
                value = index % N
                pred_label = logit[key][value].argmax(-1)

                if pred_label == 0:
                    continue
                    
                rel = {}
                rel["head_id"] = key_index[key] 
                rel["head"] = (entities[b]["start"][rel["head_id"]], entities[b]["end"][rel["head_id"]])
                rel["head_type"] = entities[b]["label"][rel["head_id"]]

                rel["tail_id"] = value_index[value]
                rel["tail"] = (entities[b]["start"][rel["tail_id"]], entities[b]["end"][rel["tail_id"]])
                rel["tail_type"] = entities[b]["label"][rel["tail_id"]]
                rel["type"] = 1
              
                pred_relations.append(rel)
            
            all_pred_relations.append(pred_relations)
        return all_pred_relations 
            
    def get_loss_1(self,l_logits,t_logits,entities,relations,key_mask,value_mask):
        loss_fcn = CrossEntropyLoss()
        B = l_logits.shape[0]
        device = l_logits.device
        loss = 0
        all_layout_logits = []
        all_text_logits = []
        all_labels = []
        key_mask = key_mask.to(torch.bool)
        value_mask = value_mask.to(torch.bool)
        for b in range(B):
            l_logit = l_logits[b][key_mask[b]]
            l_logit = l_logit[:,value_mask[b]]
            t_logit = t_logits[b][key_mask[b]]
            t_logit = t_logit[:,value_mask[b]]
            M,N,_ = l_logit.shape 

            key_index,value_index = self.devide_entities(entities[b])
            key_list = relations[b]['head']
            value_list = relations[b]['tail']
            labels = torch.zeros(M*N,device=device).view(M,N)

            true_label = []
            for i in range(len(key_list)):
                try:
                    key = key_index.index(key_list[i])
                    value = value_index.index(value_list[i])
                    labels[key][value] = 1
                    true_label.append((key*N+value))
                except:
                    continue
            
            labels = labels.view(-1).to(dtype=torch.long)
            layout_logit = l_logit.view(M*N,-1).to(dtype=torch.float)
            text_logit = t_logit.view(M*N,-1).to(dtype=torch.float)

            all_layout_logits.append(layout_logit)
            all_text_logits.append(text_logit)
            all_labels.append(labels)

        all_layout_logits = torch.cat(all_layout_logits,0)
        all_text_logits = torch.cat(all_text_logits,0)
        all_labels = torch.cat(all_labels,0)

        layout_loss = loss_fcn(all_layout_logits+1e-10,all_labels)
        text_loss = loss_fcn(all_text_logits+1e-10,all_labels)

        loss = 2*layout_loss + text_loss 
        return loss 
    
    def get_predicted_relations_1(self,l_logits,t_logits,entities,relations,key_mask,value_mask):
        #from IPython import embed;embed()
        softmax = nn.Softmax(dim=-1)
        all_pred_relations = []
      
        device = l_logits.device
        B = l_logits.shape[0]
      

        key_mask = key_mask.to(torch.bool)
        value_mask = value_mask.to(torch.bool)
        for b in range(B):
            pred_relations = []
            l_logit = l_logits[b][key_mask[b]]
            l_logit = l_logit[:,value_mask[b]]
            t_logit = t_logits[b][key_mask[b]]
            t_logit = t_logit[:,value_mask[b]]
            M,N,_ = l_logit.shape
        
            key_index,value_index = self.devide_entities(entities[b])
            key_list = relations[b]['head']
            value_list = relations[b]['tail']
            labels = torch.zeros(M*N,device=device).view(M,N)

            for index in range(M*N):
                key = index // N
                value = index % N
                layout_pred_label = l_logit[key][value].argmax(-1)
                text_pred_label = t_logit[key][value].argmax(-1)

                if layout_pred_label * text_pred_label == 0:
                    continue
                    
                rel = {}
                rel["head_id"] = key_index[key] 
                rel["head"] = (entities[b]["start"][rel["head_id"]], entities[b]["end"][rel["head_id"]])
                rel["head_type"] = entities[b]["label"][rel["head_id"]]

                rel["tail_id"] = value_index[value]
                rel["tail"] = (entities[b]["start"][rel["tail_id"]], entities[b]["end"][rel["tail_id"]])
                rel["tail_type"] = entities[b]["label"][rel["tail_id"]]
                rel["type"] = 1
              
                pred_relations.append(rel)
            
            all_pred_relations.append(pred_relations)
        return all_pred_relations 
            










