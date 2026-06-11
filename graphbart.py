import os
import pickle
import sys
import random
from collections import namedtuple
import torch
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.utils import *
from utils.model_utils import GraphBARTMultiGPUWrapper
import torch
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv
ENTITY_TYPE = ['<Task>', '<Material>', '<Method>', '<Metric>', '<Generic>', '<OtherScientificTerm>']

RELATION_TYPE = ['--ROOT--', '--PART-OF--', '--PART-OF--_INV', '--USED-FOR--',
                 '--USED-FOR--_INV', '--COMPARE--', '--COMPARE--_INV',
                 '--FEATURE-OF--', '--FEATURE-OF--_INV', '--HYPONYM-OF--',
                 '--HYPONYM-OF--_INV', '--EVALUATE-FOR--', '--EVALUATE-FOR--_INV',
                 '--CONJUNCTION--', '--CONJUNCTION--_INV']


RELATION_DESCRIPTION = [
    'title placeholder', 'is part of', 'has component', 'is used for', 'uses', 'compare to', 'compare to',
    'is feature of', 'has feature', 'is hyponym of', 'is superordinate of', 'is used to evaluate for',
    'is evaluated by', 'is conjunction of', 'is conjunction of'
]

InputData = namedtuple('InputData', [
    'paper_id', 'src_tokens', 'tgt_tokens', 'entity_type', 'node_data', 'graph',
    'ref_emb', 'citation_emb','entity_num'
])
"""
paper_id
src_tokens: tokenized oracle
tgt_tokens: tokenized review
entity_type
node_data
graph
ref_emb
citation_emb
entity_num
"""


class GraphBART:
    def __init__(self, args):
        """
        :param device: cuda or cpu
        :param knowledge: Choose from ref_graph, citation_graph and concept graph
        """
        self.args = args

        assert args.gpu in [0, 1, 2, 3]
        self._device = 'cuda' if args.gpu > 0 else 'cpu'

        self._src_max_length = args.src_max_length
        self._tgt_max_length = args.tgt_max_length

        # For knowledge
        self._knowledge = []
        if args.ref_graph:
            self._knowledge.append('ref')
            print(f'Using reference graph knowledge.')
        if args.citation_graph:
            self._knowledge.append('citation')
            print(f'Using citation graph knowledge.')
        if args.concept_graph:
            self._knowledge.append('concept')
            print(f'Using concept graph knowledge.')

        self._graphbart = GraphBARTMultiGPUWrapper(args)
        self._config = self._graphbart.config

        # May need 4 different optimizers for (1) bart (2) knowledge cross attn (3) graph (4) other embedding stuff
        self._optimizer1 = None
        self._optimizer2 = None
        self._optimizer3 = None
        self._optimizer4 = None
        self._lr_scheduler1 = None
        self._lr_scheduler2 = None
        self._lr_scheduler3 = None
        self._lr_scheduler4 = None

        self._dataset = {}

        # For calculating loss
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self._graphbart.config.pad_token_id
        )

        # Encode all the relations
        self.rel_type_token = []
        for each in RELATION_DESCRIPTION:
            self.rel_type_token.append(self._graphbart.encode(each, self._src_max_length))
        print(f'Encoded relations...')

    def get_optimizer(self, lr1, lr2, lr3, lr4, train_steps, warmup_steps,
                      weight_decay, adam_epsilon):
        no_decay = ["bias", "LayerNorm.weight"]
        bart_out = ["ent_attn", "ent_layer_norm"]
        # optimizer 1 use for pretrained BART
        p1 = [
            {"params": [p for n, p in self._graphbart.interface.named_parameters()
                        if not any(nd in n for nd in no_decay + bart_out)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in self._graphbart.interface.named_parameters()
                        if (any(nd in n for nd in no_decay)
                            and not any(bo in n for bo in bart_out))],
             "weight_decay": 0.0}
        ]

        # optimizer 2 use for from scratch training
        p2 = [
            {"params": [p for n, p in self._graphbart.ent_type_emb.named_parameters()]
                       + [p for n, p in self._graphbart.rel_emb.named_parameters()]
                       + [p for n, p in self._graphbart.gtrans.named_parameters()
                          if not any(nd in n for nd in no_decay)]
                       + [p for n, p in self._graphbart.ent_dim_redc.named_parameters()
                          if not any(nd in n for nd in no_decay)]
                       + [p for n, p in self._graphbart.gtrans_inc.named_parameters()
                          if not any(nd in n for nd in no_decay)]
                       + [p for n, p in self._graphbart.gtrans_dec.named_parameters()
                          if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in self._graphbart.gtrans.named_parameters()
                        if any(nd in n for nd in no_decay)]
                       + [p for n, p in self._graphbart.ent_dim_redc.named_parameters()
                          if any(nd in n for nd in no_decay)]
                       + [p for n, p in self._graphbart.gtrans_inc.named_parameters()
                          if any(nd in n for nd in no_decay)]
                       + [p for n, p in self._graphbart.gtrans_dec.named_parameters()
                          if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}
        ]

        # optimizer 3 use for knowledge cross attn in bart
        # a matrix and layernorm
        p3 = [
            {"params": [p for n, p in self._graphbart.interface.named_parameters()
                        if (not any(nd in n for nd in no_decay) and any(bo in n for bo in bart_out))],
             "weight_decay": weight_decay},
            {"params": [p for n, p in self._graphbart.interface.named_parameters()
                        if any(nd in n for nd in no_decay) and any(bo in n for bo in bart_out)],
             "weight_decay": 0.0}
        ]

        # for reference embedding and citation embedding
        p4 = [
            {"params": [p for n, p in self._graphbart.ref_prepend.named_parameters()
                        if not any(nd in n for nd in no_decay)]
                       + [p for n, p in self._graphbart.ref_redc.named_parameters()
                          if not any(nd in n for nd in no_decay)]
                       + [p for n, p in self._graphbart.cit_prepend.named_parameters()
                          if not any(nd in n for nd in no_decay)]
                       + [p for n, p in self._graphbart.cit_redc.named_parameters()
                          if not any(nd in n for nd in no_decay)]
                       + [p for n, p in self._graphbart.ref_cit_redc.named_parameters()
                          if not any(nd in n for nd in no_decay)]
                       + [p for n, p in self._graphbart.ref_cit_prepend.named_parameters()
                          if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in self._graphbart.ref_prepend.named_parameters()
                        if any(nd in n for nd in no_decay)]
                       + [p for n, p in self._graphbart.ref_redc.named_parameters()
                          if any(nd in n for nd in no_decay)]
                       + [p for n, p in self._graphbart.cit_prepend.named_parameters()
                          if any(nd in n for nd in no_decay)]
                       + [p for n, p in self._graphbart.cit_redc.named_parameters()
                          if any(nd in n for nd in no_decay)]
                       + [p for n, p in self._graphbart.ref_cit_redc.named_parameters()
                          if any(nd in n for nd in no_decay)]
                       + [p for n, p in self._graphbart.ref_cit_prepend.named_parameters()
                          if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}
        ]

        self._optimizer1 = AdamW(p1, lr=lr1, eps=adam_epsilon)
        self._optimizer2 = AdamW(p2, lr=lr2, eps=adam_epsilon)
        self._optimizer3 = AdamW(p3, lr=lr3, eps=adam_epsilon)
        self._optimizer4 = AdamW(p4, lr=lr4, eps=adam_epsilon)

        self._lr_scheduler1 = get_linear_schedule_with_warmup(
            self._optimizer1, num_warmup_steps=warmup_steps,
            num_training_steps=train_steps)

        self._lr_scheduler2 = get_linear_schedule_with_warmup(
            self._optimizer2, num_warmup_steps=warmup_steps,
            num_training_steps=train_steps)

        self._lr_scheduler3 = get_linear_schedule_with_warmup(
            self._optimizer3, num_warmup_steps=warmup_steps,
            num_training_steps=train_steps
        )

        self._lr_scheduler4 = get_linear_schedule_with_warmup(
            self._optimizer4, num_warmup_steps=warmup_steps,
            num_training_steps=train_steps
        )

    def save_model(self, path):
        torch.save(self._graphbart.state_dict(), path)
        print(f'Model saved in {path}')

    def load_model(self, path):
        trained_dic = torch.load(path, map_location=self._device)
        model_dict = self._graphbart.state_dict()
        # Filter out unnecessary keys
        trained_dic = {k: v for k, v in trained_dic.items() if k in model_dict}

        model_dict.update(trained_dic)
        self._graphbart.load_state_dict(model_dict)
        print(f'Model {path} loaded.')
        # self._graphbart.load_state_dict(torch.load(path, map_location=self._device))
        # print(f'Model {path} loaded.')

    def load_data(self, set_type, data):
        """ Tokenize the input. data is List[Datum]"""
        # Load data from pkl file if exists
        cache_dir = self.args.cache_dir
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        assert os.path.exists(cache_dir) and os.path.isdir(cache_dir)

        file_path = os.path.join(cache_dir, f'data_{self.args.source}_{set_type}.pkl')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                self._dataset[set_type] = pickle.load(f)
                print(f'Loading {set_type} data from {file_path}')
                print(f'#{set_type}: {len(self._dataset[set_type])}')
            return

        # If has not been cached, read the data
        self._dataset[set_type] = []
        #_dataset是一个空词典
        for datum in tqdm(data, total=len(data), desc=f'Loading {set_type} data...'):
            paper_id = datum.paper_id
            if self.args.source == 'intro':
                source = datum.raw_intro
            elif self.args.source == 'ext':
                source = datum.raw_ext
            elif self.args.source == 'abs_ext':
                source = datum.raw_abs_ext
            else:
                source = datum.raw_oracle
            src_tokens = self._graphbart.encode(source, self._src_max_length)
            tgt_tokens = self._graphbart.encode(datum.raw_review, self._tgt_max_length)
            entity_type = torch.tensor([ENTITY_TYPE.index(x) for x in datum.raw_types], dtype=torch.long)
            #将文字的类型转为了数字[2,4,0,5,5]这样的tensor
            # with open('source.pkl','wb') as f:
            #     pickle.dump([source,datum.raw_review,src_tokens,tgt_tokens,entity_type],f)
            #     print("cnt1")

            # TODO: combine entity_tokens and relation tokens together
            entity_tokens = [self._graphbart.encode(ent, self._src_max_length) for ent in datum.raw_entities]
            #长为25得list，里面全是数字
            #Input all nodes into encoder
            rel_data = ['--ROOT--'] + sum([[x[1], x[1] + '_INV'] for x in datum.raw_relations], [])
            #rel_data的结果：就是一个list，这里x[1]是取出'--USED-FOR--'，然后再这个增加了一个同样的但有“_INV”的元素
            #长度为len（raw_relations）*2+1
            #至于sum()操作是这样的处理
            #对一个列表[[1,2,3]]
            #如果sum([[1,2,3]],[]),结果是[1,2,3],可以理解成降维度
            # ['--ROOT--',
            #  '--USED-FOR--',
            #  '--USED-FOR--_INV',
            #  '--EVALUATE-FOR--',
            #  '--EVALUATE-FOR--_INV',
            #  '--EVALUATE-FOR--',
            #  '--EVALUATE-FOR--_INV',
            #  '--USED-FOR--',
            #  '--USED-FOR--_INV',
            #  '--USED-FOR--',
            #  '--USED-FOR--_INV']
            rel_tokens = [self._graphbart.encode(datum.raw_title, self._src_max_length)] \
                       + [self.rel_type_token[RELATION_TYPE.index(x)] for x in rel_data[1:]]
            #这个就是把rel_data中除了root以外得都给token化，然后root那里用实际得title来encoder
 #            #[tensor([    0, 46101,  6645, 32024,  1725,  2716,  7987,  3658,    13,  2548,
 #            12,  1116,    12, 42390, 35719, 42516, 38091,     2]),#这个对应的是title encoder的结果
 #              tensor([  0, 354, 341,  13,   2]),#其他的就是对获取每个relation的encoder的结果（可以看看rel-type_token)
 #                  tensor([   0, 9764,    2]),
 #              tensor([    0,   354,   341,     7, 10516,    13,     2]),
 #                  tensor([    0,   354, 15423,    30,     2])
            #rel_tokens = [self.rel_type_token[RELATION_TYPE.index(x)] for x in rel_data[1:]]
            # with open("rel_data.pkl",'wb') as f:
            #     pickele.dump([entity_tokens,rel_data,slf._graphbart.encode(datum.raw_title, self._src_max_length),rel_tokens],f)
            node_data = collate_tokens(entity_tokens + rel_tokens, pad_idx=self._graphbart.config.pad_token_id)
            #shape:[54,18],做一个pad得操作，固定长度18，不够得就填1
            #pickle.dump([node_data,entity_tokens,rel_tokens,rel_data,datum.raw_relations],open('middle/node_data.pkl','wb'))
            #self.len
            #将entity和rel的tokens合在一起做一个pad（也就是让特征维度一样）,长度是他两长度相加
            graph = datum.graph
            ref_emb = torch.tensor([datum.ref_emb], dtype=torch.float)
            citation_emb = torch.tensor([datum.citation_emb], dtype=torch.float)
            entity_num=datum.entity_num
            self._dataset[set_type].append(InputData(
                paper_id=paper_id,
                src_tokens=src_tokens.unsqueeze(0),
                tgt_tokens=tgt_tokens.unsqueeze(0),#成为二维
                entity_type=entity_type,#tensor([2, 4, 0, 5, 5, 5, 0, 2, 4, 5, 5, 2, 2, 2, 5, 5, 5, 5, 5, 5, 4, 0, 1, 2,
        #1])
                node_data=node_data,
                graph=graph,
                ref_emb=ref_emb,
                citation_emb=citation_emb,
                entity_num=entity_num#修改的
            ))

        print(f'#{set_type}: {len(self._dataset[set_type])}')
        with open(file_path, 'wb') as f:
            pickle.dump(self._dataset[set_type], f)
        # with open('abc.pkl', 'wb') as f:
        #     pickle.dump([paper_id,src_tokens.unsqueeze(0),tgt_tokens.unsqueeze(0),entity_type,node_data], f)
        # print("cnt2")
        # Write file to disk if it's the first time
        # with open("_dataset.pkl", 'wb') as f:
        #     pickle.dump(self._dataset[set_type], f)
        # with open(file_path, 'wb') as f:
        #     pickle.dump(self._dataset[set_type], f)
        # print("cnt3")

    def train_epoch(self, batch_size):
        assert 'train' in self._dataset
        print('start train')
        random.shuffle(self._dataset['train'])
        running_loss = 0.
        running_step = 0
        with trange(0, len(self._dataset['train']), batch_size, desc='GraphBART Training') as idxs:
            for i in idxs:
                self._graphbart.set_mode('train')
                self._graphbart.train()

                batch = self._dataset['train'][i: i + batch_size]
                self._optimizer1.zero_grad()
                self._optimizer2.zero_grad()
                self._optimizer3.zero_grad()
                self._optimizer4.zero_grad()

                # We access the data one by one
                for j in range(0, len(batch)):
                    data = batch[j]
                    # pickle.dump([batch,data],open("Graph in training",'wb'))
                    loss = self._get_seq2seq_loss(data)
                    running_loss += loss.item()
                    running_step += 1
                    loss = loss / batch_size
                    loss.backward()
                    idxs.set_postfix({'loss': running_loss / running_step}, refresh=False)

                self._optimizer1.step()
                self._optimizer2.step()
                self._optimizer3.step()
                self._optimizer4.step()
                self._lr_scheduler1.step()
                self._lr_scheduler2.step()
                self._lr_scheduler3.step()
                self._lr_scheduler4.step()


    def evaluate(self):
        assert 'val' in self._dataset
        self._graphbart.set_mode('train')
        self._graphbart.eval()
        print('start evaluate')
        loss_list = []
        for i in tqdm(range(0, len(self._dataset['val']))):
            data = self._dataset['val'][i]

            with torch.no_grad():
                loss = self._get_seq2seq_loss(data)
            loss_list.append(loss.item())
        return sum(loss_list) / len(loss_list)

    def generate(self, data):
        self._graphbart.set_mode('infer')
        self._graphbart.eval()
        # pickle.dump([data,self.args.max_len,self.args.min_len,self.args.beam,
        #              self.args.lenpen,self.args.no_repeat_ngram_size],open('middle/gen_input.pkl','wb'))
        output = self._graphbart.generate(
            data=data,
            max_length=self.args.max_len,#1024
            min_length=self.args.min_len,#100
            num_beams=self.args.beam,#4
            length_penalty=self.args.lenpen,#2
            no_repeat_ngram_size=self.args.no_repeat_ngram_size#3
        )
        return output

    def _get_seq2seq_loss(self, data):

        src_tokens = data.src_tokens
        tgt_tokens = data.tgt_tokens
        entity_type = data.entity_type
        node_data = data.node_data
        graph = data.graph
        ref_emb = data.ref_emb
        citation_emb = data.citation_emb
        entity_num=data.entity_num
        #pickle.dump([data,src_tokens,tgt_tokens, entity_type,node_data,graph],open("Graph in get_seq2seq",'wb'))
        logits = self._graphbart(
            src_tokens=src_tokens,
            tgt_tokens=tgt_tokens,
            entity_type=entity_type,
            node_data=node_data,
            graph=graph,
            ref_emb=ref_emb,
            citation_emb=citation_emb,
            entity_num = entity_num
        )
        tgt_tokens = tgt_tokens.to(logits.device)
        # with open("logit.pkl", 'wb') as f:
        #     pickle.dump([logits,tgt_tokens], f)
        # Shift so that tokens < n predict n
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = tgt_tokens[:, 1:].contiguous()

        # Flatten the tokens
        loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss

    @property
    def dataset(self):
        return self._dataset

    @property
    def train_dataset(self):
        return self._dataset['train']
