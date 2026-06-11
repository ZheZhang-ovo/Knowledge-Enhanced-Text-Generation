# Adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bart.py
import pickle
import random
from copy import deepcopy
import torch.nn.functional as F
import torch
import transformers
from .generation_utils import *
from transformers import BartTokenizer, BartForConditionalGeneration
#from transformers.generation import BeamHypotheses, BeamSearchScorer

from .modules import *


class GraphBARTMultiGPUWrapper(nn.Module):

    def __init__(self, args):
        super(GraphBARTMultiGPUWrapper, self).__init__()

        assert args.gpu in [0, 1, 2, 3]
        if args.gpu > 0:
            assert torch.cuda.is_available() and torch.cuda.device_count() > 0
            self._device = 'cuda'
        else:
            self._device = 'cpu'
        print(f'Using {self._device}...')

        self.args = args

        # We shard the model into multiple gpus if possible
        self._device_encoder = None
        self._device_decoder1 = self._device_decoder2 = None

        # BART
        self.interface = BartForConditionalGeneration.from_pretrained(args.checkpoint)
        self._tokenizer = BartTokenizer.from_pretrained(args.checkpoint)

        # Reference graph layer initialization
        # For prepend
        self.ref_prepend = nn.Linear(args.emb_dim, args.hidden_size)
        self.ref_prepend.weight.data.normal_(mean=0.0, std=0.02)
        self.ref_prepend.bias.data.zero_()
        # For reduction after concatenation
        self.ref_redc = nn.Linear(args.emb_dim + args.hidden_size, args.hidden_size)
        self.ref_redc.weight.data.normal_(mean=0.0, std=0.02)
        self.ref_redc.bias.data.zero_()

        # Citation graph layer initialization
        # For prepend
        self.cit_prepend = nn.Linear(args.emb_dim, args.hidden_size)
        self.cit_prepend.weight.data.normal_(mean=0.0, std=0.02)
        self.cit_prepend.bias.data.zero_()
        # For reduction after concatenation
        self.cit_redc = nn.Linear(args.emb_dim + args.hidden_size, args.hidden_size)
        self.cit_redc.weight.data.normal_(mean=0.0, std=0.02)
        self.cit_redc.bias.data.zero_()

        self.ref_cit_redc = nn.Linear(args.emb_dim * 2 + args.hidden_size, args.hidden_size)
        self.ref_cit_redc.weight.data.normal_(mean=0.0, std=0.02)
        self.ref_cit_redc.bias.data.zero_()

        # For prepend together ref and cit
        self.ref_cit_prepend = nn.Linear(args.emb_dim * 2, args.hidden_size)
        self.ref_cit_prepend.weight.data.normal_(mean=0.0, std=0.02)
        self.ref_cit_prepend.weight.data.zero_()

        # Concept graph layers initialization
        self.ent_type_emb = nn.Embedding(args.ent_type_num, args.hidden_size)
        nn.init.xavier_normal_(self.ent_type_emb.weight)
        self.rel_emb = nn.Embedding(args.rel_type_num, args.graph_hidden_size)
        nn.init.xavier_normal_(self.rel_emb.weight)
        # Used for dimension reduction after concatenating entity type embedding
        # and entity text embedding
        self.ent_dim_redc = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.ent_dim_redc.weight.data.normal_(mean=0.0, std=0.02)
        self.ent_dim_redc.bias.data.zero_()
        self.gtrans = GraphTransformer(args)
        # Used for dimension decrease before GAT
        self.gtrans_dec = nn.Linear(args.hidden_size, args.graph_hidden_size)
        # Used for dimension increase after GAT
        self.gtrans_inc = nn.Linear(args.graph_hidden_size, args.hidden_size)

        # Add knowledge cross attention layer and a layer norm for each BartDecoderLayer
        for i in range(len(self.encoder.layers)):
            self.encoder.layers[i].ent_attn = Attention(args)
            self.encoder.layers[i].ent_layer_norm = deepcopy(self.encoder.layers[i].self_attn_layer_norm)
            if args.knowledge_first:
                self.encoder.layers[i].knowledge_first = True
            else:
                self.encoder.layers[i].knowledge_first = False

        self._mode = None

    def set_mode(self, mode):
        assert mode in ['train', 'infer']

        if self._mode == mode:
            return

        if self.args.gpu == 3:
            if mode == 'train':
                assert torch.cuda.device_count() >= 3
                # make sure it has enough resources
                self._device_encoder = 'cuda:0'
                self._device_decoder1 = 'cuda:1'
                self._device_decoder2 = 'cuda:2'
            else:
                # During inference we only use one GPU
                self._device_encoder = self._device_decoder1 = self._device_decoder2 = 'cuda:0'
            self.cuda()
        elif self.args.gpu == 2:
            if mode == 'train':
                assert torch.cuda.device_count() >= 2
                self._device_encoder = 'cuda:0'
                self._device_decoder1 = self._device_decoder2 = 'cuda:1'
            else:
                # either only have 1 GPU, or during inference
                self._device_encoder = self._device_decoder1 = self._device_decoder2 = 'cuda:0'
            self.cuda()
        elif self._device == 'cuda':
            self._device_encoder = self._device_decoder1 = self._device_decoder2 = 'cuda:0'
            self.cuda()
        else:
            self._device_encoder = self._device_decoder1 = self._device_decoder2 = self._device

        # Model Sharding
        self.encoder.to(self._device_encoder)
        self.decoder.to(self._device_decoder1)

        # We shard the second half of decoder into another gpu if possible
        decoder_layer_num = len(self.decoder.layers)
        for i in range(decoder_layer_num):
            if i >= (decoder_layer_num // 2):
                self.decoder.layers[i].to(self._device_decoder2)
        # if self.decoder.layer_norm:
        #     self.decoder.layer_norm.to(self._device_decoder2)
        if self.decoder.layernorm_embedding:
            self.decoder.layernorm_embedding.to(self._device_decoder2)

        # For calculating lm logits
        self.interface.final_logits_bias = move_device(
            self.interface.final_logits_bias, self._device_decoder2)
        self.model.shared = move_device(self.model.shared, self._device_decoder2)

        # Reference graph layers
        self.ref_prepend.to(self._device_encoder)
        self.ref_redc.to(self._device_encoder)

        # Citation graph layers
        self.cit_prepend.to(self._device_encoder)
        self.cit_redc.to(self._device_encoder)

        self.ref_cit_redc.to(self._device_encoder)
        self.ref_cit_prepend.to(self._device_encoder)

        # concept graph layers
        self.ent_type_emb.to(self._device_encoder)
        self.rel_emb.to(self._device_encoder)
        self.ent_dim_redc.to(self._device_encoder)
        self.gtrans.to(self._device_encoder)
        self.gtrans_inc.to(self._device_encoder)
        self.gtrans_dec.to(self._device_encoder)

        torch.cuda.empty_cache()

        # Set mode
        if mode == 'train':
            self.train()
        else:
            self.eval()

        self._mode = mode
        self.decoder.mode = mode  # use for decide whether to cache previous states

    def encode(self, sentence, max_length):
        """ Encode text (up to max_length)
            Example output:
            tensor([0, 9226, 16, 41, 15162, 2])
        """
        tokens = self._tokenizer([sentence], max_length=max_length,
                                 truncation=True, return_tensors='pt')['input_ids'][0].tolist()

        new_tokens = [elem for elem in tokens if self.is_valid(elem)]
        return torch.tensor(new_tokens).long()

    def is_valid(self, element: int):
        """ In the vocab: 50264 is <mask>, 50265 is None, we need to avoid 50264 and 50265 """
        if element != 50264 and element != 50265:
            return True
        else:
            return False

    def forward(self, src_tokens, tgt_tokens,  node_data, graph,
               entity_num,decoder_inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,return_dict=None):
        decoder_input_ids = tgt_tokens
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if src_tokens is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                src_tokens, self.config.pad_token_id, self.config.decoder_start_token_id
            )
        if self.training:
            use_cache = False
        else:
            use_cache = True
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = False
        """ Forward text directly """
        src_tokens = src_tokens.to(self._device_encoder)
        attention_mask = src_tokens.ne(self.config.pad_token_id)
        ent_out, g_root = self.get_ent_out(node_data, graph, entity_num)
        if not self.args.concept_graph:
            ent_out = None
        # Get representations of source tokens
        encoder_out = forward_encoder(
            self=self.encoder,
            src_tokens=src_tokens,
            ent_out=ent_out,
            attention_mask=attention_mask
        )  # (1 X 27 X 1024)
        encoder_out = encoder_out[0]

        #pickle.dump([src_tokens,attention_mask,encoder_out,ref_emb,citation_emb,g_root,ent_out,self.config.pad_token_id],open("middle/enrich.pkl",'wb'))
        #Injecting reference embedding and citation embedding as well as global node of concept graph
        # encoder_out, src_tokens, attention_mask = self.enrich_encoder_out(#这里的1都是batchsize
        #     src_tokens=src_tokens,#shape:[1,969]就是把文章token化
        #     attention_mask=attention_mask,#config.pad_token_id这个等于1，shape:[1,969]，里面的元素全是True或者False
        #     encoder_out=encoder_out,#shape:[1,969,1024]就是把token向量化
        #     # ref_emb=ref_emb,#shape[1,128]
        #     # citation_emb=citation_emb,#shape[1,128]
        #     global_node=g_root,#shape:[1024]root也就是title的embedding
        #     entity_nodes=ent_out#shape[1,25,1024]25就是entity个数
        # )
        #原先3个encoder_out是[1, 969, 1024]变成了[1, 970, 1024]，也是增加了一个citation_emb的信息
        #Remove ent_out
        # if self.args.prepend_concept:
        #     ent_out = None

        # decoder cached states
        #prev_output_tokens是target_tokens,#这是的attention_mask是对source的mask
        outputs = self.get_decoder_out(input_ids=decoder_input_ids, encoder_out=encoder_out,
                                    encoder_attention_mask=attention_mask,use_cache=use_cache,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states)

        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.interface.final_logits_bias)

        return lm_logits

    def enrich_encoder_out(self, src_tokens, attention_mask, encoder_out, ref_emb, citation_emb, global_node,
                           entity_nodes):
        """ Use reference embedding and citation embedding to refine encoder out """
        # common for prepend
        prepend_tokens = torch.tensor([self.config.bos_token_id]).unsqueeze(1).to(src_tokens.device)
        #这个是tensor（[[0]]),device='cuda',shape是【1，1】
        prepend_mask = torch.tensor([True]).unsqueeze(1).to(attention_mask.device)
        #是tensor（[[True]]),device='cuda',shape是【1，1】
        #pickle.dump([prepend_tokens,prepend_mask,self.config.bos_token_id],open("middle/enrich_middle.pkl",'wb'))
        if self.args.concept_graph_global:#默认False
            # only prepend to the encoder out
            global_node = global_node.unsqueeze(1).to(encoder_out.device)
            encoder_out = torch.cat([global_node, encoder_out], dim=1)
            src_tokens = torch.cat([prepend_tokens, src_tokens], dim=1)
            attention_mask = torch.cat([prepend_mask, attention_mask], dim=1)
            #print(1)

        if self.args.prepend_concept:#默认False
            entity_num = entity_nodes.shape[1]
            entity_nodes = entity_nodes.to(encoder_out.device)
            entity_prepend_tokens = torch.tensor([self.config.bos_token_id] * entity_num) \
                .unsqueeze(0).to(src_tokens.device)
            entity_prepend_mask = torch.tensor([True] * entity_num).unsqueeze(0).to(attention_mask.device)
            encoder_out = torch.cat([entity_nodes, encoder_out], dim=1)
            src_tokens = torch.cat([entity_prepend_tokens, src_tokens], dim=1)
            attention_mask = torch.cat([entity_prepend_mask, attention_mask], dim=1)
            #print(2)

        # If we use reference embedding information
        if self.args.ref_graph:#默认False
            if self.args.citation_graph and self.args.prepend:
                # ref embedding + cit embedding
                ref_emb = ref_emb.to(self.ref_cit_prepend.weight.device)
                citation_emb = citation_emb.to(self.ref_cit_prepend.weight.device)
                together_emb = torch.cat([ref_emb, citation_emb], dim=1)
                together_transformed = self.ref_cit_prepend(together_emb).unsqueeze(1)
                encoder_out = torch.cat([together_transformed, encoder_out], dim=1)
                src_tokens = torch.cat([prepend_tokens, src_tokens], dim=1)
                attention_mask = torch.cat([prepend_mask, attention_mask], dim=1)
                #print(3)
            elif self.args.prepend:
                # transform the embedding and concate it with encoder_out
                ref_emb = ref_emb.to(self.ref_prepend.weight.device)
                ref_emb_transformed = self.ref_prepend(ref_emb).unsqueeze(1)

                # prepend the reference graph for encoder_out
                encoder_out = torch.cat([ref_emb_transformed, encoder_out], dim=1)
                # prepend the reference graph for src_tokens
                src_tokens = torch.cat([prepend_tokens, src_tokens], dim=1)
                # prepend the attention mask
                attention_mask = torch.cat([prepend_mask, attention_mask], dim=1)
               # print(4)
            else:
                # concate the embedding with encoder out, then pass a dimension reduction matrix
                ref_emb = ref_emb.unsqueeze(1)
                ref_emb = ref_emb.expand(1, encoder_out.shape[1], ref_emb.shape[-1])
                ref_emb = ref_emb.to(encoder_out.device)
                encoder_out = torch.cat([ref_emb, encoder_out], dim=-1)
                #print(5)

        # If we use citation embedding information
        if self.args.citation_graph:#默认True
            if self.args.ref_graph and self.args.prepend:
                #print(6)
                pass
            elif self.args.prepend:#默认进入这里，引入外部知识
                citation_emb = citation_emb.to(self.cit_prepend.weight.device)
                citation_emb_transformed = self.cit_prepend(citation_emb).unsqueeze(1)
                #将【1，128】的tensor变成了【1，1，1024】的tensor
                encoder_out = torch.cat([citation_emb_transformed, encoder_out], dim=1)
                #原先encoder_out是[1,969,1024]变成了[1,970,1024]
                src_tokens = torch.cat([prepend_tokens, src_tokens], dim=1)
                # 原先encoder_out是[1,969,1024]变成了[1,970,1024]
                attention_mask = torch.cat([prepend_mask, attention_mask], dim=1)
                # 原先encoder_out是[1,969,1024]变成了[1,970,1024]
                #这是的1都是batchsize
                #pickle.dump([citation_emb,citation_emb_transformed,encoder_out,src_tokens,attention_mask], open("middle/ci.pkl", 'wb'))
                #print(7)
            else:
                citation_emb = citation_emb.unsqueeze(1)
                citation_emb = citation_emb.expand(1, encoder_out.shape[1], citation_emb.shape[-1])
                citation_emb = citation_emb.to(encoder_out.device)
                encoder_out = torch.cat([citation_emb, encoder_out], dim=-1)
                #print(8)

        # Use different dimension reduction matrix#默认后面3个都不进去
        if self.args.ref_graph and not self.args.citation_graph and not self.args.prepend:
            encoder_out = self.ref_redc(encoder_out)
            #print(9)
        if not self.args.ref_graph and self.args.citation_graph and not self.args.prepend:
            encoder_out = self.cit_redc(encoder_out)
            #print(10)
        if self.args.ref_graph and self.args.citation_graph and not self.args.prepend:
            encoder_out = self.ref_cit_redc(encoder_out)
           # print(11)
        #pickle.dump([encoder_out, src_tokens, attention_mask],open('middle/return.pkl','wb'))
        #返回值没有变化
        return encoder_out, src_tokens, attention_mask

    def get_ent_out(self, node_data, graph,entity_num):
        """ Get entity representations by going through GraphTransformer """
        ent_out, g_root = None, None
        if self.args.concept_graph or self.args.concept_graph_global:
            # Get initial node representations
            raw_node_enc = forward_encoder(
                self=self.encoder,
                src_tokens=node_data,
                attention_mask=node_data.ne(self.config.pad_token_id),#node data中元素值不等于self.config.pad_token_id返回True，其他返回false
            )
            raw_node_enc=raw_node_enc[0]
            '''
             #将（54，18）这样的全是index组成的tensor，转成了（54，18，1024）这样的tensor（元素都是小数）
            #54，总节点数（entity数25，关系数14*2,加上一个root）
            #18是pad的长度
            #1024是隐藏层维度
            '''

            # Take the representation at EOS to be the node representation
            eos_mask = node_data.eq(self.config.eos_token_id)#node data中元素值等于self.config.eos_token_id返回True，其他返回false
            '''
            #返回诸如下面的
            # tensor([[False, False, False, True, False, False, False, False, False, False,
            #          False, False, False, False, False, False, False, False],
            #         [False, False, True, False, False, False, False, False, False, False,
            #          False, False, False, False, False, False, False, False],
            #产生原因
            #对第一个tensor，其第一个非填充的是第3个元素（从0开始算），所以True是第三个
            #对第二个tensor同理
            # tensor([[0, 15238, 1743, 2, 1, 1, 1, 1, 1, 1,
            #          1, 1, 1, 1, 1, 1, 1, 1],
            #         [0, 243, 2, 1, 1, 1, 1, 1, 1, 1,
            #          1, 1, 1, 1, 1, 1, 1, 1]
            '''

            node_enc = raw_node_enc[eos_mask, :].view(raw_node_enc.size(0), -1,
                                                      raw_node_enc.size(-1))[:, -1, :]

            '''
             #node_enc的shape就是（54，1024），也就是获得了每个点的initial state
             #实际上raw_node_enc[eos_mask, :]与上面的效果一样
            '''
            # print("ABCCCCCCCC")
            # with open("rawdata.pkl", 'wb') as f:
            #     pickle.dump([raw_node_enc, node_data,eos_mask], f)
            #pickle.dump([node_data,raw_node_enc,self.config.eos_token_id,eos_mask,node_enc],open("Graph in ent_out.pkl",'wb'))
            # Decrease the dimension
            node_enc = self.gtrans_dec(node_enc)#降到200维

            # Go through Graph Transformer
            graph = graph.to(self._device_encoder)
            #print("OKKKKKK")
            # with open("data.pkl",'wb') as f:
            #     pickle.dump([graph,node_enc],f)
            #这里的graph只是edge_index
            ent_out, g_root = self.gtrans(graph, node_enc,entity_num)
            '''
            #ent_out:(25,200)也就是只返回entity_node的embedding
            #g_root：（1，200）也就是返回root的embedding
            '''
            # pickle.dump([node_enc,ent_out, g_root],
            #             open("Graph after GAT.pkl", 'wb'))
            # Increase the dimension
            g_root = self.gtrans_inc(g_root)
            ent_out = self.gtrans_inc(ent_out)
            ent_out = ent_out.unsqueeze(0)  # (1 x 18 x 1024)
        return ent_out, g_root

    def get_decoder_out(self,  input_ids, encoder_out,
                        encoder_attention_mask,
                        output_attentions=False,use_cache=None,output_hidden_states=None,
                        return_dict=False):
        # """ Given encoder outputs, decoder input, get decoder outputs """
        # encoder_ent_mask = None
        # if ent_out is not None:
        #     encoder_ent_mask = torch.tensor([True] * ent_out.shape[1]).expand(ent_out.shape[0],
        #                                                                       ent_out.shape[1])
        # #encoder_ent_mask,shape是[1,25]然后全是True
        decoder_attention_mask = input_ids.ne(self.config.pad_token_id)
        # outputs = forward_decoder(
        #     self=self.decoder,
        #     input_ids=input_ids,
        #     attention_mask=decoder_attention_mask,
        #     encoder_hidden_states=encoder_out,
        #     encoder_attention_mask=encoder_attention_mask,
        #     head_mask=None,
        #     cross_attn_head_mask=None,
        #     past_key_values=None,
        #     inputs_embeds=None,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict
        # )
        current_device=self.decoder.device
        input_ids=move_device(input_ids,current_device)
        decoder_attention_mask=move_device(decoder_attention_mask,current_device)
        encoder_out=move_device(encoder_out,current_device)
        encoder_attention_mask=move_device(encoder_attention_mask,current_device)
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_out,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        return outputs

    def generate(self, data, max_length, min_length,
                 num_beams, length_penalty, no_repeat_ngram_size):
        output = generate(
            self=self,
            data=data,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        return output

    @property
    def config(self):
        return self.interface.model.config

    @property
    def model(self):
        return self.interface.model

    @property
    def encoder(self):
        return self.interface.model.encoder

    @property
    def decoder(self):
        return self.interface.model.decoder

    @property
    def tokenizer(self):
        return self._tokenizer


def forward_embedding(self, tokens):
    """ Embed the tokens """
    inputs_embeds = self.embed_tokens(tokens.to(self.embed_tokens.weight.device)) \
                    * self.embed_scale

    embed_pos = self.embed_positions(tokens.to(self.embed_positions.weight.device))

    inputs_embeds = move_device(inputs_embeds, embed_pos.device)
    x = inputs_embeds + embed_pos

    x = move_device(x, self.layernorm_embedding.weight.device)
    x = self.layernorm_embedding(x)

    x = F.dropout(x, p=self.dropout, training=self.training)
    return x

def _expand_mask(mask, dtype, tgt_len = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask#取反

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
def forward_encoder_layer(self,hidden_states,attention_mask,ent_out=None,
                          layer_head_mask=None,
                          output_attentions = False):
    """
           Args:
               self:代表encoder_layer
               hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
               attention_mask (`torch.FloatTensor`): attention mask of size
                   `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
               layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                   `(encoder_attention_heads,)`.
               output_attentions (`bool`, *optional*):
                   Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                   returned tensors for more detail.
           """
    # attention:[1,1,969,969]全是0，代表不mask
    # pickle.dump([ attention_mask],
    #             open("G:/GEAR/KK/KIDReview-main/KIDReview-main/encoder_atten_layer_middle.pkl", 'wb'))
    layer_state = {}
    encoder_ent_mask=None
    if ent_out is not None:
        encoder_ent_mask = torch.tensor([True] * ent_out.shape[1]).expand(ent_out.shape[0],
                                                                          ent_out.shape[1])
        encoder_ent_mask = invert_mask(encoder_ent_mask)
    residual = hidden_states
    if self.knowledge_first:
        if ent_out is not None:
            hidden_states=attend_to_ent(self,hidden_states,ent_out,
                                    encoder_ent_mask=encoder_ent_mask,layer_state=layer_state,
                                    output_attentions=output_attentions)
    hidden_states, attn_weights, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
    )
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = residual + hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)
    if not self.knowledge_first:
        if ent_out is not None:
            hidden_states=attend_to_ent(self,hidden_states,ent_out,
                                    encoder_ent_mask=encoder_ent_mask, layer_state=layer_state,
                                    output_attentions=output_attentions
                                    )
    residual = hidden_states
    hidden_states = self.activation_fn(self.fc1(hidden_states))
    hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states = self.fc2(hidden_states)
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = residual + hidden_states
    hidden_states = self.final_layer_norm(hidden_states)

    if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
    ):
        clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs
def forward_encoder(self, src_tokens, ent_out=None,attention_mask=None, head_mask=None,
                    output_attentions=False,src_embeds=None,
                    output_hidden_states=False, return_dict=False):
    r"""
           Args:
               self is bartencoder
               input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                   Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                   provide it.

                   Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                   [`PreTrainedTokenizer.__call__`] for details.

                   [What are input IDs?](../glossary#input-ids)
               attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                   Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                   - 1 for tokens that are **not masked**,
                   - 0 for tokens that are **masked**.

                   [What are attention masks?](../glossary#attention-mask)
               head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                   Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                   - 1 indicates the head is **not masked**,
                   - 0 indicates the head is **masked**.

               inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                   Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                   This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                   than the model's internal embedding lookup matrix.
               output_attentions (`bool`, *optional*):
                   Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                   returned tensors for more detail.
               output_hidden_states (`bool`, *optional*):
                   Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                   for more detail.
               return_dict (`bool`, *optional*):
                   Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
           """
    attention_mask=attention_mask.long()#将全False或True的转为全0或者1
    #丢进encoder中的，如果是0就是不mask，1就是mask
    #这里attention_mask就是全1
    #pickle.dump([src_tokens,attention_mask],open("G:/GEAR/KK/KIDReview-main/KIDReview-main/encoder_atten.pkl", 'wb'))
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    input_ids=src_tokens
    inputs_embeds=src_embeds
    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input = input_ids
        input_ids = input_ids.view(-1, input_ids.shape[-1])
    elif inputs_embeds is not None:
        input = inputs_embeds[:, :, -1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids.to(self.embed_tokens.weight.device)) * self.embed_scale

    embed_pos = self.embed_positions(input)
    embed_pos = embed_pos.to(inputs_embeds.device)

    hidden_states = inputs_embeds + embed_pos
    hidden_states=move_device(hidden_states,self.layernorm_embedding.weight.device)
    hidden_states = self.layernorm_embedding(hidden_states)
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

    # expand attention_mask
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)
        attention_mask=move_device(attention_mask,hidden_states.device)
    # 这里attention_mask就是全0了，然后丢进去就是不mask
    #pickle.dump([ attention_mask], open("G:/GEAR/KK/KIDReview-main/KIDReview-main/encoder_atten_input.pkl", 'wb'))
    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    # check if head_mask has a correct number of layers specified if desired
    if head_mask is not None:
        if head_mask.size()[0] != (len(self.layers)):
            raise ValueError(
                f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                f" {head_mask.size()[0]}."
            )

    for idx, encoder_layer in enumerate(self.layers):
        current_device=encoder_layer.fc1.weight.device
        hidden_states=move_device(hidden_states,current_device)
        attention_mask=move_device(attention_mask,current_device)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        dropout_probability = random.uniform(0, 1)
        if self.training and (dropout_probability < self.layerdrop):  # skip the layer
            layer_outputs = (None, None)
        else:
            layer_outputs = forward_encoder_layer(
                self=encoder_layer,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                ent_out=ent_out,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
    return hidden_states, encoder_states, all_attentions
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
def _prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
        ).to(inputs_embeds.device)

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask
def forward_decoder(
        self,
        input_ids = None,
        attention_mask= None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask= None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache= None,
        output_attentions= None,
        output_hidden_states =None,
        return_dict=None
):
    """
    Includes several features from "Jointly Learning to Align and
    Translate with Transformer Models" (Garg et al., EMNLP 2019).

    Args:
        self: In the model, self is self.decoder
        tgt_tokens (LongTensor): previous decoder outputs of shape
            `(batch, tgt_len)`, for teacher forcing
        encoder_hidden_states/ent_out: output from the encoder, used for
            encoder-side attention
        encoder_padding_mask/encoder_ent_mask: for ignoring pad tokens
        decoder_cached_states (dict or None): dictionary used for storing state during generation

    Returns:
        tuple:
            - the decoder's features of shape `(batch, tgt_len, embed_dim)`
            - hidden states
            - attentions
    """
    use_cache = (self.mode == 'infer')
    output_attentions = False
    output_hidden_states = False
    return_dict = False
    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        input = input_ids
        input_shape = input.shape
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        input = inputs_embeds[:, :, -1]
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input.to(self.embed_tokens.weight.device)) * self.embed_scale
        inputs_embeds = inputs_embeds.to(self.embed_tokens.weight.device)
    attention_mask = _prepare_decoder_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_key_values_length
    )
    # attention_mask.shape[1,1,694,694]
    # decoder_padding_mask[1,694]
    # pickle.dump([attention_mask,decoder_padding_mask],open('decoder_attention.pkl','wb'))
    # x = self.embed_tokens(input_ids.to(self.embed_tokens.weight.device)) * self.embed_scale
    if encoder_hidden_states is not None and encoder_attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
    # encoder_attention_mask.shape【1，1，694，969】
    positions = self.embed_positions(input_ids.to(self.embed_positions.weight.device))
    inputs_embeds = inputs_embeds.to(positions.device)
    hidden_states = inputs_embeds + positions
    #print(hidden_states.shape)#[1,694,1024]
    hidden_states = self.layernorm_embedding(hidden_states.to(self.layernorm_embedding.weight.device))
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    # pickle.dump([tgt_tokens.cpu(),inputs_embeds.cpu(),hidden_states.cpu(),encoder_attention_mask],open('forward_decoder.pkl','wb'))
    # tgt_tokens.shape[1,694]
    # inputs_embeds.shape[1,694,1024]
    # hidden_states.shape[1,694,1024]
    #print(hidden_states.shape)#[1,694,1024]
    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
    next_decoder_cache = () if use_cache else None
    for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
        if attn_mask is not None:
            if attn_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )
    # print(hidden_states.shape)#[1,694,1024]
    for idx, decoder_layer in enumerate(self.layers):
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        current_device = decoder_layer.fc1.weight.device

        hidden_states = move_device(hidden_states, current_device)
        encoder_attention_mask = move_device(encoder_attention_mask, current_device)
        attention_mask = move_device(attention_mask, current_device)
        encoder_hidden_states = move_device(encoder_hidden_states, current_device)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        dropout_probability = random.uniform(0, 1)
        if self.training and (dropout_probability < self.layerdrop):
            continue

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = forward_decoder_layer(
                self=decoder_layer,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                ),
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

            if encoder_hidden_states is not None:
                all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    return tuple(
        v
        for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
        if v is not None
    )

def forward_decoder_layer(
    self,
    hidden_states,
    attention_mask= None,
    encoder_hidden_states = None,
    encoder_attention_mask = None,
    layer_head_mask= None,
    cross_attn_layer_head_mask = None,
    past_key_value= None,
    output_attentions= False,
    use_cache = None,
) :
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        encoder_hidden_states (`torch.FloatTensor`):
            cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
        encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
            `(encoder_attention_heads,)`.
        cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
            size `(decoder_attention_heads,)`.
        past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
    """
    residual = hidden_states
    # pickle.dump([hidden_states,attention_mask,
    # encoder_hidden_states,
    # encoder_attention_mask ,
    # layer_head_mask,
    # cross_attn_layer_head_mask ,
    # past_key_value,
    # output_attentions,
    # use_cache ],open("middle/修改/decoder_layer_input.pkl",'wb'))
    # Self Attention
    # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
    self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
    #self_attn_past_key_value=None
    # add present self-attn cache to positions 1,2 of present_key_value tuple
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        past_key_value=self_attn_past_key_value,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
    )
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = residual + hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)

    # Cross-Attention Block
    cross_attn_present_key_value = None
    cross_attn_weights = None
    if encoder_hidden_states is not None:
        residual = hidden_states

        # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
        cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=cross_attn_past_key_value,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # add cross-attn to positions 3,4 of present_key_value tuple
        present_key_value = present_key_value + cross_attn_present_key_value

    # Fully Connected
    residual = hidden_states
    hidden_states = self.activation_fn(self.fc1(hidden_states))
    hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states = self.fc2(hidden_states)
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = residual + hidden_states
    hidden_states = self.final_layer_norm(hidden_states)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights, cross_attn_weights)

    if use_cache:
        outputs += (present_key_value,)

    return outputs
# def forward_decoder_layer(self, x, encoder_hidden_states, ent_out, encoder_attention_mask=None,
#                           encoder_ent_mask=None, layer_state=None, causal_mask=None,use_cache=None,
#                            past_key_value=None,output_attentions=False):
#     """ Output cross attention weights """
#     if layer_state is None:
#         layer_state = {}
#
#     # Self Attention
#     residual = x
#     self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
#     #print(x.shape)#[694,1,1024]
#     hidden_states, self_attn_weights, present_key_value = self.self_attn(
#         hidden_states=x,
#         past_key_value=self_attn_past_key_value,
#         attention_mask=causal_mask,
#         output_attentions=output_attentions
#     )
#
#     hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
#     hidden_states = residual + hidden_states
#     hidden_states = self.self_attn_layer_norm(hidden_states)
#
#     if self.knowledge_first:
#         # Cross attention for knowledge
#         if ent_out is not None:
#             hidden_states = attend_to_ent(self, hidden_states, ent_out, encoder_ent_mask, layer_state,
#                                           output_attentions)
#
#     # Cross attention for text
#     cross_attn_present_key_value = None
#     cross_attn_weights = None
#     if encoder_hidden_states is not None:
#         residual = hidden_states
#     cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
#     hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
#         hidden_states=hidden_states,
#         key_value_states=encoder_hidden_states,
#         attention_mask=encoder_attention_mask,
#         past_key_value=cross_attn_past_key_value,
#         output_attentions=output_attentions,
#     )
#     hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
#     hidden_states = residual + hidden_states
#     hidden_states = self.encoder_attn_layer_norm(hidden_states)
#     present_key_value = present_key_value + cross_attn_present_key_value
#     if not self.knowledge_first:
#         # knowledge_first=False，走这条
#         # Cross attention for knowledge
#         # Pop out some cache in {}
#         if ent_out is not None:
#             hidden_states = attend_to_ent(self, hidden_states, ent_out, encoder_ent_mask, layer_state,
#                                           output_attentions)
#
#     # Fully Connected
#     residual = hidden_states
#     hidden_states = self.activation_fn(self.fc1(hidden_states))
#     hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
#     hidden_states = self.fc2(hidden_states)
#     hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
#     hidden_states = residual + hidden_states
#     hidden_states = self.final_layer_norm(hidden_states)
#     outputs = (hidden_states,)
#
#     if output_attentions:
#         outputs += (self_attn_weights, cross_attn_weights)
#
#     if use_cache:
#         outputs += (present_key_value,)
#
#     return outputs


def attend_to_ent(self, x, ent_out, encoder_ent_mask, layer_state, output_attentions):
    """ Cross attention to entities in decoder layer
        self: decoder layer
    """
    residual = x
    x = self.self_attn_layer_norm(x)

    ent_out = ent_out.to(x.device)
    encoder_ent_mask = encoder_ent_mask.to(x.device)
    x, _ = self.ent_attn(
        query=x,
        key=ent_out,
        key_padding_mask=encoder_ent_mask,
        layer_state=layer_state,
        output_attentions=output_attentions
    )
    x = nn.functional.dropout(x, p=self.dropout, training=self.training)
    x=x.transpose(0,1)#将【694，1，1024】变成【1，694，1024】
    x = residual + x
    x = self.self_attn_layer_norm(x)
    return x


def invert_mask(attention_mask):
    """Turns 1->0, 0->1, False->True, True-> False"""
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)


def move_device(tensor, device):
    if tensor is None:
        return None
    else:
        tensor = tensor.to(device)
        return tensor


def _prepare_bart_decoder_inputs(config, input_ids, decoder_input_ids=None,
                                 decoder_padding_mask=None, causal_mask_dtype=torch.float32):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    pad_token_id = config.pad_token_id
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    causal_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
        dtype=causal_mask_dtype, device=decoder_input_ids.device
    )
    return decoder_input_ids, decoder_padding_mask, causal_mask


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


# ===================================================== #
#                      Generation                       #
# ===================================================== #

@torch.no_grad()
def generate(self: GraphBARTMultiGPUWrapper, data, max_length,
             min_length, num_beams, length_penalty, no_repeat_ngram_size,
             **model_specific_kwargs):
    """
    Return token ids (LongTensor)
    data=data,
    max_length=self.args.max_len,#1024
    min_length=self.args.min_len,#100
    num_beams=self.args.beam,#4
    length_penalty=self.args.lenpen,#2
    no_repeat_ngram_size=self.args.no_repeat_ngram_size#3
    """

    encoder_device = self.encoder.embed_tokens.weight.device
    input_ids = data.src_tokens#已经是token了
    input_ids = input_ids.to(encoder_device)
    # batch_size = input_ids.shape[0]
    # original_attention_mask = input_ids.ne(self.config.pad_token_id).long()
    #
    # effective_batch_size = batch_size
    # effective_batch_mult = 1
    #
    # # Here the input_ids is used to get the encoder output
    # original_encoder_outputs: tuple = self.encoder(input_ids, attention_mask=original_attention_mask)
    # encoder_outputs = original_encoder_outputs[0]
    # #ent_out, g_root = self.get_ent_out(data.node_data, data.graph)
    # ent_out, g_root = self.get_ent_out(data.node_data, data.graph,data.entity_num)
    # encoder_outputs, input_ids, original_attention_mask = self.enrich_encoder_out(
    #     #     src_tokens=input_ids,
    #     #     attention_mask=original_attention_mask,
    #     encoder_out=encoder_outputs,
    #     ref_emb=data.ref_emb,
    #     citation_emb=data.citation_emb,
    #     global_node=g_root,
    #     entity_nodes=ent_out
    # )
    # pickle.dump([ent_out, g_root,encoder_outputs,input_ids,original_attention_mask,], open('middle/generate/get_ent_out.pkl', 'wb'))
    # if self.args.prepend_concept:
    #     ent_out = None
    #
    # input_ids_len = input_ids.shape[-1]
    #
    # attention_mask = original_attention_mask.unsqueeze(1).expand(
    #     batch_size, effective_batch_mult * num_beams, input_ids_len
    # )
    #
    # attention_mask = attention_mask.contiguous().view(
    #     effective_batch_size * num_beams, input_ids_len
    # )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
    # # used for masking attention to the encoder side
    #
    # # create empty decoder_input_ids
    # input_ids = torch.full(
    #     (effective_batch_size * num_beams, 1),
    #     self.config.decoder_start_token_id,
    #     dtype=torch.long,
    #     device=next(self.interface.parameters()).device,
    # )
    # cur_len = 1
    #
    # # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and
    # # num_return_sequences > 1)
    # expanded_batch_idxs = (
    #     torch.arange(batch_size)
    #         .view(-1, 1)
    #         .repeat(1, num_beams * effective_batch_mult)
    #         .view(-1)
    #         .to(input_ids.device)
    # )
    #
    # # expand encoder_outputs
    # encoder_outputs = (encoder_outputs.index_select(0, expanded_batch_idxs),
    #                    *original_encoder_outputs[1:])  # (1 X 14 X 1024) --> (4 X 14 X 1024)
    #
    # ent_out = ent_out.index_select(0,
    #                                expanded_batch_idxs) if ent_out is not None else None  # (1 X 17 X 1024) --> (4 X 17 X 1024)
    #
    # output = _generate_beam_search(
    #     self=self,
    #     input_ids=input_ids,
    #     cur_len=cur_len,
    #     max_length=max_length,
    #     min_length=min_length,
    #     no_repeat_ngram_size=no_repeat_ngram_size,
    #     batch_size=effective_batch_size,
    #     length_penalty=length_penalty,
    #     num_beams=num_beams,
    #     encoder_outputs=encoder_outputs,
    #     ent_out=ent_out,
    #     attention_mask=attention_mask,
    #     model_specific_kwargs=model_specific_kwargs,
    # )
    output=self.interface.generate(input_ids,max_length=max_length,min_length=min_length
                                   ,num_beams=num_beams,length_penalty=length_penalty,
                                   no_repeat_ngram_size=no_repeat_ngram_size)
    # # decode the generated text (and then encode in order to align predition)
    # decoded_text = [self.tokenizer.decode(g, skip_special_tokens=True,
    #                                       clean_up_tokenization_spaces=False).strip()
    #                 for g in output]
    decoded_text=self._tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return decoded_text
# def calc_banned_ngram_tokens(prev_input_ids, num_hypos, no_repeat_ngram_size, cur_len) :
#     """Copied from fairseq for no_repeat_ngram in beam_search"""
#     if cur_len + 1 < no_repeat_ngram_size:
#         # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
#         return [[] for _ in range(num_hypos)]
#     generated_ngrams = [{} for _ in range(num_hypos)]
#     for idx in range(num_hypos):
#         gen_tokens = prev_input_ids[idx].tolist()
#         generated_ngram = generated_ngrams[idx]
#         for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
#             prev_ngram_tuple = tuple(ngram[:-1])
#             generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
#
#     def _get_generated_ngrams(hypo_idx):
#         # Before decoding the next token, prevent decoding of ngrams that have already appeared
#         start_idx = cur_len + 1 - no_repeat_ngram_size
#         ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
#         return generated_ngrams[hypo_idx].get(ngram_idx, [])
#
#     banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
#     return banned_tokens
# def calc_banned_bad_words_ids(prev_input_ids, bad_words_ids) :
#     banned_tokens = []
#
#     def _tokens_match(prev_tokens, tokens):
#         if len(tokens) == 0:
#             # if bad word tokens is just one token always ban it
#             return True
#         if len(tokens) > len(prev_tokens):
#             # if bad word tokens are longer than prev tokens they can't be equal
#             return False
#
#         if prev_tokens[-len(tokens) :] == tokens:
#             # if tokens match
#             return True
#         else:
#             return False
#
#     for prev_input_ids_slice in prev_input_ids:
#         banned_tokens_slice = []
#
#         for banned_token_seq in bad_words_ids:
#             assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
#                 bad_words_ids
#             )
#
#             if _tokens_match(prev_input_ids_slice, banned_token_seq[:-1]) is False:
#                 # if tokens do not match continue
#                 continue
#
#             banned_tokens_slice.append(banned_token_seq[-1])
#
#         banned_tokens.append(banned_tokens_slice)
#
#     return banned_tokens


def set_scores_to_inf_for_banned_tokens(scores, banned_tokens) :
    """Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be
    a list of list of banned tokens to ban in the format [[batch index, vocabulary position],...]
        Args:
            scores: logits distribution of shape (batch size, vocabulary size)
            banned_tokens: list of list of tokens to ban of length (batch_size)
    """
    banned_mask_list = []
    for idx, batch_banned_tokens in enumerate(banned_tokens):
        for token in batch_banned_tokens:
            banned_mask_list.append([idx, token])
    if not banned_mask_list:
        return
    banned_mask = torch.LongTensor(banned_mask_list)
    indices = torch.ones(len(banned_mask))
    # A sparse tensor is generated from a list of coordinates: [[0, 1], [0, 2], [2, 0]]. A conversion to dense tensor generates:
    # [ 0  1  1 ]
    # [ 0  0  0 ]
    # [ 1  0  0 ]

    banned_mask = torch.sparse.LongTensor(banned_mask.t(), indices, scores.size()).to(scores.device).to_dense().bool()
    scores.masked_fill_(banned_mask, -float("inf"))
# def postprocess_next_token_scores(
#         scores,
#         input_ids,
#         no_repeat_ngram_size,
#         bad_words_ids,
#         cur_len,
#         min_length,
#         max_length,
#         eos_token_id,
#         repetition_penalty,
#         batch_size,
#         num_beams,
# ):
#     # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
#     if repetition_penalty != 1.0:
#         enforce_repetition_penalty_(
#             scores,
#             batch_size,
#             num_beams,
#             input_ids,
#             repetition_penalty,
#         )
#
#     # set eos token prob to zero if min_length is not reached
#     if eos_token_id is not None and cur_len < min_length:
#         scores[:, eos_token_id] = -float("inf")
#
#     if no_repeat_ngram_size > 0:
#         # calculate a list of banned tokens to prevent repetitively generating the same ngrams
#         num_batch_hypotheses = batch_size * num_beams
#         # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
#         banned_batch_tokens = calc_banned_ngram_tokens(
#             input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
#         )
#         for i, banned_tokens in enumerate(banned_batch_tokens):
#             scores[i, banned_tokens] = -float("inf")
#     if bad_words_ids is not None:
#         # Exclude EOS token (already processed)
#         bad_words_ids = list(filter(lambda bad_token_seq: bad_token_seq != [eos_token_id], bad_words_ids))
#         # calculate a list of banned tokens according to bad words
#         banned_tokens = calc_banned_bad_words_ids(input_ids.tolist(), bad_words_ids)
#         # Modify the scores in place by setting the banned tokens logits to `-inf`
#         set_scores_to_inf_for_banned_tokens(scores, banned_tokens)
#
#
#
#     return scores
def _generate_beam_search(self: GraphBARTMultiGPUWrapper, input_ids, cur_len, max_length, min_length,
                          no_repeat_ngram_size, batch_size, length_penalty, num_beams, encoder_outputs,
                          ent_out, attention_mask, model_specific_kwargs):
    """ Generate sequences for each example with beam search.
    """
    # Configuration
    early_stopping = self.config.early_stopping
    use_cache = self.config.use_cache
    bad_words_ids = self.config.bad_words_ids
    eos_token_id = self.config.eos_token_id
    pad_token_id = self.config.pad_token_id
    repetition_penalty = self.config.repetition_penalty
    vocab_size = self.config.vocab_size
    num_return_sequences = self.config.num_return_sequences
    # generated hypotheses
    #transformers.BeamH
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
        #BeamHypotheses(num_beams, length_penalty, early_stopping=early_stopping)
        for _ in range(batch_size)
    ]

    # scores for each sentence in the beam
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

    # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the
    # exact same tokens three times
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

    # The default cache compute states
    past = (encoder_outputs, None) if encoder_outputs is not None else None

    # Our cache compute states
    decoder_cached_states = None

    # done sentences
    done = [False for _ in range(batch_size)]

    while cur_len < max_length:
        if ent_out is not None:
            x, decoder_cached_states = self.get_decoder_out(None, input_ids, encoder_outputs[0],
                                                            attention_mask, ent_out=ent_out,
                                                            decoder_cached_states=decoder_cached_states)

            lm_logits = F.linear(x, self.model.shared.weight, bias=self.interface.final_logits_bias)
            next_token_logits = lm_logits[:, -1, :]
        else:
            # this prepare inputs for generation is different than the one above
            model_inputs = self.interface.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache,
                encoder_outputs=encoder_outputs, **model_specific_kwargs
            )

            outputs = self.interface(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
            next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

            # if model has past, then set the past variable to speed up decoding
            if to_use_cache(self, outputs, use_cache):
                past = outputs[1]

        next_token_logits = self.interface.adjust_logits_during_generation(
            next_token_logits, cur_len=cur_len, max_length=max_length
        )

        scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
        scores = postprocess_next_token_scores(
            scores=scores,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=cur_len,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=num_beams,
        )

        # We don't do sample
        next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

        # re-organize to group the beam together (we are keeping top hypothesis accross beams)
        next_scores = next_scores.view(
            batch_size, num_beams * vocab_size
        )  # (batch_size, num_beams * vocab_size)

        next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

        # next batch beam content
        next_batch_beam = []

        # for each sentence
        for batch_idx in range(batch_size):

            # if we are done with this sentence, add a pad token
            if done[batch_idx]:
                assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                assert (
                        eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue

            # next sentence beam content, this will get added to next_batch_beam
            next_sent_beam = []

            # next tokens for this sentence
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                # get beam and token IDs
                beam_id = beam_token_id // vocab_size  # the beam id within this sentence
                token_id = beam_token_id % vocab_size

                effective_beam_id = batch_idx * num_beams + beam_id
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    generated_hyps[batch_idx].add(
                        input_ids[effective_beam_id].clone(), beam_token_score.item(),
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                # once the beam for next step is full, don't add more tokens to it.
                if len(next_sent_beam) == num_beams:
                    break

            # Check if we are done so that we can save a pad step if all(done)
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

            # update next beam content
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"

        # stop when we are done with each sentence
        if all(done):
            break

        # sanity check / prepare next batch
        assert len(next_batch_beam) == batch_size * num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])

        # re-order batch and update current length
        input_ids = input_ids[beam_idx, :]
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
        cur_len = cur_len + 1

        # re-order internal states
        # TODO: Reorder decoder cache
        if ent_out is None:
            if past is not None:
                past = self.interface._reorder_cache(past, beam_idx)
        else:
            tmp_past = ((None, None), decoder_cached_states)
            decoder_cached_states = self.interface._reorder_cache(tmp_past, beam_idx)[1]

    # finalize all open beam hypotheses and add to generated hypotheses
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue

        # test that beam scores match previously calculated scores if not eos and batch_idx not done
        if eos_token_id is not None and all(
                (token_id % vocab_size).item() != eos_token_id for token_id in next_tokens[batch_idx]
        ):
            assert torch.all(
                next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
            ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx],
            )

        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)

    # depending on whether greedy generation is wanted or not define different output_batch_size and
    # output_num_return_sequences_per_batch
    output_batch_size = batch_size * num_return_sequences
    output_num_return_sequences_per_batch = num_return_sequences

    # select the best hypotheses
    sent_lengths = input_ids.new(output_batch_size)
    best = []

    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_hyp = sorted_hyps.pop()[1]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)

    # shorter batches are padded
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, "`Pad_token_id` has to be defined"
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

        # fill with hypothesis and eos_token_id if necessary
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
    else:
        # none of the hypotheses have an eos_token
        assert (len(hypo) == max_length for hypo in best)
        decoded = torch.stack(best).type(torch.long).to(next(self.interface.parameters()).device)

    return decoded
