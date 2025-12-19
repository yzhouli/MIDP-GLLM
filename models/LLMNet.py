import os

import torch

from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig

from models.Graph_LLM import GraphLLM
from models.Graph_LLM_Deepseek import GraphLLM_Deepseek
from models.Graph_LLM_GPT2 import GraphLLM_GPT2
from models.Graph_LLM_Llama import GraphLLM_Llama
from models.Graph_LLM_MobileLLM import GraphLLM_MobileLLM


def load_pretrained_weights(custom_layer, pretrained_layer):
    pretrained_dict = pretrained_layer.state_dict()
    custom_dict = custom_layer.state_dict()
    weight_dict = dict()
    for k, v in custom_dict.items():
        if k not in pretrained_dict:
            raise RuntimeError(f'{k} not in pretrained model')
        weight_dict[k] = pretrained_dict[k]
    custom_layer.load_state_dict(weight_dict, strict=False)
    return custom_layer


@torch.no_grad()
def based_model(qwen_weight_path, layer_num, graph_layer_num, save_weight_path, LLM_Class):
    device = torch.device("cpu")
    if not os.path.exists(save_weight_path):
        # 配置参数
        config = AutoConfig.from_pretrained(
            qwen_weight_path,
            trust_remote_code=True
        )
        config.use_cache = False

        qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_weight_path,
            config=config,
            trust_remote_code=True
        ).to(device)
        qwen_model.config.num_hidden_layers = layer_num
        qwen_llm = LLM_Class(config=qwen_model.config, graph_layer_num=graph_layer_num).to(device)

        try:
            pred_trained_model = qwen_model.model  # Qwen3
            qwen_llm = load_pretrained_weights(qwen_llm, pred_trained_model)
        except:
            pred_trained_model = qwen_model.base_model  # GPT2
            qwen_llm = load_pretrained_weights(qwen_llm, pred_trained_model)
        checkpoint = {
            'model_state': qwen_llm.state_dict(),
            'config': qwen_llm.config
        }
        torch.save(checkpoint, save_weight_path)
        for k, v in qwen_llm.state_dict().items():
            print(f'{k}')
        raise exit(f'LLM weight is not saved, please re-run!')
    checkpoint = torch.load(save_weight_path, weights_only=False)
    config = checkpoint['config']
    if config.num_hidden_layers != layer_num:
        raise RuntimeError(
            f'Pre-trained Qwen model layer num does not match ({config.num_hidden_layers} != {layer_num})!')
    graph_llm = LLM_Class(config, graph_layer_num=graph_layer_num).to(device)
    graph_llm.load_state_dict(checkpoint['model_state'])
    return graph_llm


class GPTUtil(object):
    @staticmethod
    @torch.no_grad()
    def based_model(qwen_weight_path, layer_num, graph_layer_num, save_weight_path):
        return based_model(qwen_weight_path, layer_num, graph_layer_num, save_weight_path, GraphLLM_GPT2)

class QWENUtil(object):
    @staticmethod
    @torch.no_grad()
    def based_model(qwen_weight_path, layer_num, graph_layer_num, save_weight_path):
        return based_model(qwen_weight_path, layer_num, graph_layer_num, save_weight_path, GraphLLM)

class LLAMAUtil(object):
    @staticmethod
    @torch.no_grad()
    def based_model(qwen_weight_path, layer_num, graph_layer_num, save_weight_path):
        return based_model(qwen_weight_path, layer_num, graph_layer_num, save_weight_path, GraphLLM_Llama)

class DEEPSEEKUtil(object):
    @staticmethod
    @torch.no_grad()
    def based_model(qwen_weight_path, layer_num, graph_layer_num, save_weight_path):
        return based_model(qwen_weight_path, layer_num, graph_layer_num, save_weight_path, GraphLLM_Deepseek)

class MOILELLMUtil(object):
    @staticmethod
    @torch.no_grad()
    def based_model(qwen_weight_path, layer_num, graph_layer_num, save_weight_path):
        return based_model(qwen_weight_path, layer_num, graph_layer_num, save_weight_path, GraphLLM_MobileLLM)


def LLMUtil(llm_path):
    llm_name = llm_path.split('/')[-1].lower()
    if 'gpt_2' in llm_name:
        return GPTUtil
    elif 'wen' in llm_name:
        return QWENUtil
    elif 'llama' in llm_name:
        return LLAMAUtil
    elif 'deepseek' in llm_name:
        return DEEPSEEKUtil
    elif 'mobilellm' in llm_name:
        return MOILELLMUtil
    raise RuntimeError(f'{llm_name} not support!')

def HIDDEN_SIZE_UTIL(llm_name, llm):
    if 'gpt_2' in llm_name:
        return llm.config.hidden_size
    elif 'qwen' in llm_name:
        return llm.config.hidden_size
    elif 'llama' in llm_name:
        return llm.config.hidden_size
    elif 'deepseek' in llm_name:
        return llm.config.hidden_size
    elif 'mobilellm' in llm_name:
        return llm.config.hidden_size



class LLMNet(nn.Module):
    def __init__(self, llm, args):
        super(LLMNet, self).__init__()
        self.llm = llm
        self.args = args
        llm_name = args.llm_path.split('/')[-1].lower()
        llm_hidden_size = HIDDEN_SIZE_UTIL(llm_name, llm)
        self.input_embedding = nn.Linear(self.args.d_model, llm_hidden_size)
        self.output_linear = nn.Linear(llm_hidden_size, self.args.d_model)
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=self.args.dropout)
        self.init_weights()

    def get_qk_states(self):
        return self.llm.qk_states_li

    def forward(self, x, pos_mask=None, adj_graph_mask=None):
        x = self.input_embedding(x)
        out = self.llm(inputs_embeds=x, attention_mask=pos_mask, adj_graph_mask=adj_graph_mask,
                       output_hidden_states=True, rel_type='multi_scale_causal_adj')
        # out = self.llm(inputs_embeds=x, attention_mask=pos_mask,
        #                output_hidden_states=True)
        att_mat = out.hidden_states[-1]
        att_mat = att_mat + x
        att_mat = self.activation(att_mat)
        att_mat = self.output_linear(att_mat)
        att_mat = self.activation(att_mat)
        att_mat = self.dropout(att_mat)
        att_mat = att_mat.squeeze(0)
        return att_mat

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()
