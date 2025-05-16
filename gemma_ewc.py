
import pickle

from transformers import (
        Gemma2ForCausalLM,
        )
import torch



# May be able to generalize later for other models for causal LM?
class GemmaEWC(Gemma2ForCausalLM):
    def __init__(
            self,
            config,
            EWC_lambda, # Lol these should probably in a config class or sth
            EWC_param_dir,
            ):
        super().__init__(config)
        self.EWC_lambda = EWC_lambda
        with open(EWC_param_dir, 'rb') as f:
            mean, self.fisher = pickle.load(f)
        self.mean = {}
        for param_name, param_val in mean:
            self.mean[param_name] = param_val
        self.__EWC_clean()
        self.__EWC_to(self.device)


    def __EWC_clean(self):
        for param_name, param_val in self.fisher.items():
            inf_indices = param_val == torch.inf
            self.fisher[param_name][inf_indices] = 1e2
            param_val.requires_grad_(False) 
        for _, param in self.mean.items():
            param.requires_grad_(False) 

    def __EWC_to(self, *args, **kwargs):
        for param_name, param_val in self.mean.items():
            self.mean[param_name] = param_val.to(*args, **kwargs)
        for param_name, param_val in self.fisher.items():
            self.fisher[param_name] = param_val.to(*args, **kwargs)

    # CANT USE **SOME_KWARGS BECAUSE I HAVE TO MAINTAIN SIGNATURE?????
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        cache_position = None,
        logits_to_keep = 0,
        **loss_kwargs,
        ):
        output = super().forward(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
                cache_position,
                logits_to_keep,
                **loss_kwargs,
                )


        ewc_loss = self._compute_EWC_loss()
        output.loss += ewc_loss
        return output

    def to(
            self,
            *args,
            **kwargs,
            ):
        # There must be a better way to do this...
        self.__EWC_to(*args, **kwargs)
        super().to(*args, **kwargs)

    def _compute_EWC_loss(self):
        ewc_losses = []
        for param_name, param_val in self.named_parameters():
            mean_param = self.mean[param_name]
            distance = (param_val - mean_param) ** 2
            scaled_loss = self.fisher[param_name] * distance
            ewc_losses.append(scaled_loss.sum())
        loss_sum = sum(ewc_losses)
        ewc_loss = self.EWC_lambda/2 * loss_sum
        return ewc_loss
