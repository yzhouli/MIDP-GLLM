import numpy as np
import torch

from models.IDP_LLM import IDP_LLM
from models.LLMNet import LLMUtil, LLMNet
from utils import Constants


class IDP_LLM_LoRA(IDP_LLM):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--llm_path', type=str, default='Please enter the LLM model weight address')
        parser.add_argument('--weight_path', type=str, default='data/weight/llm_weight.pt')
        parser.add_argument('--block_layers', type=int, default=10)
        parser.add_argument('--lora_layers', type=int, default=3)
        parser.add_argument('--remove_weight', type=bool, default=True)
        parser.add_argument('--pos_dim', type=int, default=64)
        parser.add_argument('--n_heads', type=int, default=8)
        parser.add_argument('--gcn_layers', type=int, default=8)
        parser.add_argument('--ssl_reg', type=float, default=1e-3)
        parser.add_argument('--alpha', type=float, default=1.0)
        parser.add_argument('--ssl_temp', type=float, default=0.1)
        parser.add_argument('--K', type=int, default=10)
        return parser

    def lora_model(self, llm_model, lora_num):
        from peft import LoraConfig, get_peft_model
        lora_target_modules, modules = [], []
        for i in range(llm_model.config.num_hidden_layers - lora_num, llm_model.config.num_hidden_layers):
            modules.append(f'layers.{i}.self_attn.q_proj')
            modules.append(f'layers.{i}.self_attn.k_proj')
            modules.append(f'layers.{i}.self_attn.v_proj')
            modules.append(f'layers.{i}.self_attn.o_proj')
            modules.append(f'layers.{i}.Qwen3MLP.gate_proj')
            modules.append(f'layers.{i}.Qwen3MLP.up_proj')
            modules.append(f'layers.{i}.Qwen3MLP.down_proj')  # Qwen3 DeepSeek
        # for i in range(llm_model.config.num_hidden_layers - lora_num, llm_model.config.num_hidden_layers):
        #     modules.append(f'h.{i}.attn.c_attn')
        #     modules.append(f'h.{i}.attn.c_proj') #GPT2
        lora_target_modules.extend(modules)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        llm_model = get_peft_model(llm_model, lora_config)
        for k, v in llm_model.state_dict().items():
            print(f'{k}')
        return llm_model

    def __init__(self, args, data_loader):
        super(IDP_LLM_LoRA, self).__init__(args, data_loader)
        self.args = args
        self.llm_model = LLMUtil(args.llm_path).based_model(qwen_weight_path=self.args.llm_path,
                                                            layer_num=self.args.block_layers,
                                                            save_weight_path=self.args.weight_path,
                                                            graph_layer_num=self.args.lora_layers)
        if self.args.lora_layers > 0:
            self.llm_model = self.lora_model(llm_model=self.llm_model, lora_num=self.args.lora_layers)
        self.llm_layer = LLMNet(args=self.args, llm=self.llm_model)

    def forward(self, input_seq, input_timestamp, tgt_idx, rel):
        ego_embeddings = self.get_ego_embeddings().to(self.device)
        embedding_list = [ego_embeddings]
        for layer_idx in range(self.gcn_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            embedding_list.append(ego_embeddings)

        gcn_all_embeddings = torch.stack(embedding_list, dim=1)
        gcn_all_embeddings = torch.mean(gcn_all_embeddings, dim=1)

        cas_all_embeddings, user_all_embeddings = torch.split(
            gcn_all_embeddings, [self.cas_num, self.user_num]
        )

        input_seq = input_seq[:, :-1]
        mask = (input_seq == Constants.PAD)

        batch_t = torch.arange(input_seq.size(1)).expand(input_seq.size()).cuda()
        position_embed = self.pos_embedding(batch_t)

        original_seq_emb = self.user_embedding(input_seq.cuda())
        dyemb = user_all_embeddings[input_seq.cuda()]
        dyemb += position_embed

        valid_his = (input_seq > 0).long().cuda()

        attn_score = self.W2(self.W1(original_seq_emb).tanh()).cuda()
        attn_score = attn_score.masked_fill(valid_his.unsqueeze(-1) == 0, -np.inf).cuda()
        attn_score = attn_score.transpose(-1, -2).cuda()
        attn_score = (attn_score - attn_score.max()).softmax(dim=-1).cuda()
        attn_score = attn_score.masked_fill(torch.isnan(attn_score), 0).cuda()
        intention_vectors = (original_seq_emb[:, None, :, :] * attn_score[:, :, :, None]).sum(-2).sum(-2).cuda()

        intention_vectors = intention_vectors.unsqueeze(1)
        intention_vectors = intention_vectors.expand(-1, original_seq_emb.size(1), -1)

        att_out = self.align_attention(dyemb.cuda(),
                                       intention_vectors.cuda(),
                                       intention_vectors.cuda(),
                                       mask=mask.cuda())

        att_out = self.llm_layer(att_out, mask, rel) + att_out

        output = self.linear(att_out.cuda())

        mask = self.get_previous_user_mask(input_seq.cuda(), self.user_num)
        output = output.cuda() + mask.cuda()

        if self.training:
            return output.view(-1, output.size(-1)), user_all_embeddings, cas_all_embeddings, embedding_list
        else:
            return output.view(-1, output.size(-1))
