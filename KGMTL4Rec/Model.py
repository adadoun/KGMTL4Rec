import torch.nn as nn
import torch

## define ER_MLP architecture
class ER_MLP(nn.Module):
    def __init__(self, tot_entity, tot_relation, emb_size, hidden_size):
        ##
        super(ER_MLP, self).__init__()
        ##
        self.ent_embeddings = nn.Embedding(tot_entity, emb_size)
        self.rel_embeddings = nn.Embedding(tot_relation, emb_size)
        torch.normal(self.ent_embeddings.weight)
        torch.normal(self.rel_embeddings.weight)
        ##
        self.M1 = nn.Linear(emb_size, hidden_size, bias = False)
        self.M2 = nn.Linear(emb_size, hidden_size, bias = False)
        self.M3 = nn.Linear(emb_size, hidden_size, bias = False)
        ##
        self.hidden_fc = nn.Linear(hidden_size, int(hidden_size/2))
        ##
        self.hidden_fc_2 = nn.Linear(int(hidden_size/2), 1)
        #
        self.dropout = nn.Dropout(0.2)
        ##        
    def forward(self, h, r, t):
        # add hidden layer, with relu activation function
        #
        x_h = self.ent_embeddings(h)
        x_r = self.rel_embeddings(r)
        x_t = self.ent_embeddings(t)
        ##
        Tanh = torch.tanh(torch.cat(self.M1(x_h), self.M2(x_r), self.M3(x_t)),1)
        # add dropout layer
        Tanh = self.dropout(Tanh)
        #
        fc1 = self.hidden_fc(Tanh)
        #
        fc1 = torch.tanh(fc1)
        #
        fc1 = self.droput(fc1)
        #
        z = self.hidden_fc_2(fc1)
        #
        return z    

## define KGMTL architecture
class KGMTL(nn.Module):
    def __init__(self, tot_entity, tot_relation, tot_attribute, emb_size, hidden_size, word_embedding_size, n_items, CNN_out_size, CNN_stride, CNN_kernel_size):
        ##
        super(KGMTL, self).__init__()
        ## Initialize Embedding layers
        self.ent_embeddings = nn.Embedding(tot_entity, emb_size)
        self.rel_embeddings = nn.Embedding(tot_relation, emb_size)
        self.att_embeddings = nn.Embedding(tot_attribute, emb_size)
        ### Weights init
        nn.init.normal_(self.ent_embeddings.weight)
        nn.init.normal_(self.rel_embeddings.weight)
        nn.init.normal_(self.att_embeddings.weight)
        ### tail, head, relation hidden layers
        self.Mh = nn.Linear(emb_size, hidden_size, bias = False)
        self.Mr = nn.Linear(emb_size, hidden_size, bias = False)
        self.Mt = nn.Linear(emb_size, hidden_size, bias = False)
        ### hidden layer of Structnet
        self.hidden_struct_net_fc = nn.Linear(hidden_size, 1)
        ### head att, and tail att relation hidden layers
        self.ah = nn.Linear(emb_size, hidden_size, bias = False)
        self.at = nn.Linear(emb_size, hidden_size, bias = False)
        ### hidden layer of AttrNet
        self.hidden_attr_net_fc = nn.Linear(hidden_size*2, 1)
        # dropout layer (p=0.2)
        self.dropout = nn.Dropout(0.2)
        
        ## CNN Modules - DescNet
        # CNN parameters definition
        # Kernel sizes
        self.kernel_size = kernel_size
        # Output size for each convolution
        self.out_size = out_size
        # Number of strides for each convolution
        self.stride = stride

        # Embedding layer definition
        # Convolution layers definition
        self.conv_1 = nn.Conv1d(d_size, self.out_size, self.kernel_size, self.stride)
        # Max pooling layers definition
        self.pool_1 = nn.MaxPool1d(self.kernel_size, self.stride)
        # Compute output size of the Conv layer/ pool layer
        out_conv_1 = ((word_embedding_size - 1 * (self.kernel_size - 1) - 1) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1)
        out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_pool_1 = math.floor(out_pool_1)*self.out_size
        
        ### hidden layer of DescNet
        self.hidden_desc_net_fc = nn.Linear(hidden_size + out_pool_1, n_items)
        
    def StructNet_forward(self, h, r, t):
        ## 1st Part of KGMTL4REC -> StructNet
        # x_h, x_r and x_t are the embeddings 
        x_h = self.ent_embeddings(h)
        x_r = self.rel_embeddings(r)
        x_t = self.ent_embeddings(t)
        # Mh, Mr, Mt are the h,r,t hidden layers 
        ## hidden_struct_net_fc1 is the struct net hidden layer
        struct_net_fc1 = torch.tanh(self.hidden_struct_net_fc(self.Mh(x_h) + self.Mr(x_r) + self.Mt(x_t)))
        z_struct_net_fc1 = self.dropout(struct_net_fc1)
        ##
        return z_struct_net_fc1
    
    def AttrNet_h_forward(self, h, ah):
        ## 2nd part of KGMTL4REC -> AttrNet for head entity
        x_ah = self.att_embeddings(ah)
        x_h = self.ent_embeddings(h)
        ## hidden_head_att_net_fc1 is the head attribute net hidden layer
        head_att_net_fc1 = torch.tanh(self.hidden_attr_net_fc(torch.cat((self.ah(x_ah), self.Mh(x_h)),1)))
        v_head_att_net = self.dropout(head_att_net_fc1) 
        ##
        return v_head_att_net
        
    def AttrNet_t_forward(self, t, at): 
        ## 3rd part of the NN -> AttrNet for tail entity
        x_at = self.att_embeddings(at)
        x_t = self.ent_embeddings(t)
        ## hidden_head_att_net_fc1 is the head attribute net hidden layer
        tail_att_net_fc1 = torch.tanh(self.hidden_attr_net_fc(torch.cat((self.at(x_at), self.Mt(x_t)),1)))
        v_tail_att_net = self.dropout(tail_att_net_fc1)  
        ##
        return v_tail_att_net
    
    def DescNet_forward(self, e, w_d):
        ##
        # Convolution layer 1 is applied
        x_conv = self.conv_1(w_d)
        x_tanh = torch.tanh(x_conv)
        x_pool = self.pool_1(x_tanh)
        # Dropout is applied
        w_cnn_d = self.dropout(x_pool)
        ##
        x_e = self.ent_embeddings(e)
        ##
        x_soft = F.log_softmax(self.hidden_desc_net_fc(torch.tanh(torch.cat((w_cnn_d, x_e), 1))))
        ##
        return x_soft