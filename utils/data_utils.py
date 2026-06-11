#import dgl
import torch

NODE_TYPE = {'entity': 0, 'root': 1, 'relation': 2}


class Datum:
    """
    A single data point. Contains id, title, entity_text, entity_type, relation, oracle, review,
    reference embedding and citation embedding.
    """
    def __init__(self, paper_id, title, oracle,review,
                 entities, types, relations, ref_emb, citation_emb):
        self.paper_id = paper_id
        self.raw_title = title
        self.raw_oracle = oracle
        #self.raw_intro = intro
        #self.raw_ext = ext
        #self.raw_abs_ext = abs_ext
        self.raw_review = review
        self.raw_entities = entities
        self.raw_types = types.split(" ")
        self.raw_relations = []
        for relation in relations:
            #例子：loss function -- USED-FOR -- Dirichlet Prior Network
            words = relation.split(" ")
            #words:['loss', 'function', '--', 'USED-FOR', '--', 'Dirichlet', 'Prior', 'Network']
            for i in range(len(words)):
                if words[i] == '--' and words[i + 2] == '--' and words[i + 1].upper() == words[i + 1]:
                    ##找到是第一个“--”对应的I=i值
                    head = " ".join(words[:i])
                    #取出loss function
                    relation_text = words[i] + words[i + 1] + words[i + 2]
                    #--USED-FOR--
                    tail = " ".join(words[i + 3:])
                    # Make sure entities in relations are recognized
                    if head in self.raw_entities and tail in self.raw_entities:
                        self.raw_relations.append([head, relation_text, tail])
                    break
        self.ref_emb = ref_emb
        self.citation_emb = citation_emb
        #self.graph = self.build_graph()
        self.graph = self.build_graph_edge_index()
        self.entity_num=len(self.raw_entities)
    @classmethod
    def from_json(cls, json_data):
        # return cls(json_data['id'], json_data['title'], json_data['oracle'], json_data['intro'], json_data['ext'],
        #            json_data['abs_ext'], json_data['review'], json_data['entities'], json_data['types'],
        #            json_data['relations'],  json_data['ref_emb'], json_data['citation_emb'])

        return cls(json_data['id'],  json_data['title'],json_data['oracle'],  json_data['review'], json_data['entities'], json_data['types'],
               json_data['relations'], json_data['ref_emb'], json_data['citation_emb'])

    def __str__(self):
        return '\n'.join([str(k) + ":\t" + str(v) for k, v in self.__dict__.items()])

    def __len__(self):
        return len(self.raw_review)

    def build_graph_edge_index(self):
        """ Build the graph as the paper described. All relations are converted to nodes, add a global node. All
            entities are connected to the global node, all nodes (entity, relation, root) has a self-loop edge.
            For nodes,
            1. First, we add all entity nodes.
            2. Then, we add the root node.
            3. Finally, we add the relations and inverse relations nodes.
            For edges,
            1. All entity nodes are connected to the root node.
            2. All nodes has self-loop edge.
            3. Add adjacent edges.
        """
        # graph = dgl.DGLGraph()
        entity_num = len(self.raw_entities)
        relation_num = len(self.raw_relations)
        a = list(range(entity_num))
        #产生[0,1,2,3,4....,24]这样的代表entity的点
        b = list(range(entity_num + 1 + 2 * relation_num))
        #产生[0,1,2,3....53]这样代表所有的点的list
        # Add nodes
        #graph.add_nodes(entity_num, {'type': torch.ones(entity_num) * NODE_TYPE['entity']})
        #给每个entity创造节点，属性都是0
        # graph.add_nodes(1, {'type': torch.ones(1) * NODE_TYPE['root']})
        # # 创造根节点，属性是1
        # graph.add_nodes(2 * relation_num, {'type': torch.ones(2 * relation_num) * NODE_TYPE['relation']})
        # # 创造边节点，属性是2（注意×2是因为有主动和被动两种）
        # # Add edges
        # graph.add_edges(list(range(entity_num)), entity_num)
        z1 = [[x, entity_num] for x in a]
        #entity_num对应的是root的编号，这个就是让每个点都有一个指向root的边
        #graph.add_edges(entity_num, list(range(entity_num)))
        z2 = [[entity_num, x] for x in a]
        #这个是让root有指向每个entity的边
        #graph.add_edges(list(range(entity_num + 1 + 2 * relation_num)), list(range(entity_num + 1 + 2 * relation_num)))
        z3 = [[x, x] for x in b]
        #这个是给每个点产生一个自连接

        u, v = [], []

        for i, relation in enumerate(self.raw_relations):
            head_idx = self.raw_entities.index(relation[0])
            tail_idx = self.raw_entities.index(relation[2])
            rel_idx = entity_num + 1 + 2 * i
            rel_inv_idx = rel_idx + 1
            # Add head -> rel, rel -> tail, tail -> rel_inv, rel_env -> head
            u.extend([head_idx, rel_idx, tail_idx, rel_inv_idx])
            v.extend([rel_idx, tail_idx, rel_inv_idx, head_idx])
            #U是起始节点的list，V是终止节点的list
            #这个就是创建了head到rel的边，rel到tail的边。以及被动的关系
        # if len(u) > 0:
        #     graph.add_edges(u, v)
        z4 = []
        for i in range(len(u)):
            z4.append([u[i], v[i]])
        z1.extend(z2)
        z1.extend(z3)
        z1.extend(z4)
        edge_index=torch.tensor(z1,dtype=torch.long)
        return edge_index

