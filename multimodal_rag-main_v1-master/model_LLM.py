import json
import re
from typing import Optional, Any

import torch
from PIL import Image
from llama_index.core.base.llms.types import LLMMetadata, CompletionResponse, ChatResponse, CompletionResponseGen
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.callbacks import llm_completion_callback
from transformers import TextStreamer, AutoModel, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM

from LLM.MobileVLM.mobilevlm.conversation import SeparatorStyle
from LLM.SliME.llava.mm_utils import get_model_name_from_path as get_model_name_from_path_SLIME
from LLM.mipha.constants import IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN
from LLM.mipha.conversation import conv_templates
from LLM.mipha.mm_utils import tokenizer_image_token, process_images, KeywordsStoppingCriteria
from LLM.mipha.model.builder import load_pretrained_model
from LLM.MobileVLM.mobilevlm.model.mobilevlm import load_pretrained_model as load_pretrained_model_mobileVLM
from LLM.MobileVLM.mobilevlm.conversation import conv_templates as conv_templates_mobileVLM
from LLM.MobileVLM.mobilevlm.utils import process_images as process_images_mobileVLM
from LLM.MobileVLM.mobilevlm.utils import tokenizer_image_token as tokenizer_image_token_mobileVLM
from LLM.MobileVLM.mobilevlm.utils import KeywordsStoppingCriteria as KeywordsStoppingCriteria_mobileVLM
from LLM.MobileVLM.mobilevlm.utils import disable_torch_init
from LLM.mipha.serve.cli import load_image
from LLM.SliME.llava.model.builder import load_pretrained_model as load_pretrained_model_LLaVAHD
from LLM.SliME.llava.conversation import conv_templates as conv_templates_SLIME

with open('config/config.json') as user_file:
    config = user_file.read()
config = json.loads(config)
# {
#   "name": "John",
#   "age": 50,
#   "is_married": false,
#   "profession": null,
#   "hobbies": ["travelling", "photography"]
# }
DEFAULT_LLM_MODEL = config["model_LLM_path"]

class MobileVLM():
    def __init__(self, model_path=DEFAULT_LLM_MODEL):
        super().__init__()
        model_name = model_path.split('/')[-1]
        disable_torch_init()
        tokenizer, model, image_processor, context_len = load_pretrained_model_mobileVLM(model_path, False,
                                                                               False)
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len

        self.temperature = 0
        self.top_p =  None
        self.num_beams = 1
        self.max_new_tokens = 512

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        # 得到LLM的元数据
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()  # 回调函数
    def complete(self, prompt: str, image: str, **kwargs: Any) -> CompletionResponse:
        pass

    #  @llm_chat_callback()  # 回调函数
    def chat(self, prompt, **kwargs: Any) -> ChatResponse:
        # 完成函数
        prompt, image = prompt[0], prompt[1]
        conv = conv_templates_mobileVLM["v1"].copy()

        if image is not None:
            images = [Image.open(image).convert("RGB")]
            images_tensor = process_images_mobileVLM(images, self.image_processor,self.model.config).to(self.model.device, dtype=torch.float16)
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        else:
            images_tensor = None
            conv.append_message(conv.roles[0],  prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # Input
        input_ids = (
            tokenizer_image_token_mobileVLM(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())
        stopping_criteria = KeywordsStoppingCriteria_mobileVLM([stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        # Result-Decode
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        return outputs.strip()

    @llm_completion_callback()
    def stream_complete(
            self, prompt: str, image: Optional[torch.FloatTensor] = None, **kwargs: Any
    ) -> CompletionResponseGen:
        pass
class MiniCPM(CustomLLM):
    context_window: int = 8192  # 上下文窗口大小
    num_output: int = 128  # 输出的token数量
    model_name: str = "MiniCPM"  # 模型名称
    tokenizer: object = None  # 分词器
    model: object = None  # 模型
    image_processor: object = None # image_processor
    def __init__(self, pretrained_model_name_or_path=DEFAULT_LLM_MODEL):
        super().__init__()
        # GPU方式加载模型
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True,device_map="cuda:0" )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        # 得到LLM的元数据
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()  # 回调函数
    def complete(self, prompt: str, image: str, **kwargs: Any) -> CompletionResponse:
        pass

  #  @llm_chat_callback()  # 回调函数
    def chat(self, prompt, **kwargs: Any) -> ChatResponse:
        # 完成函数
        prompt, image = prompt[0], prompt[1]
        msgs = [{"content":prompt,"role":"user"}]
        if image is not None:
            image = Image.open(image).convert("RGB")
        answer = self.model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=0.7
        )
        return answer

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str,image: Optional[torch.FloatTensor] = None, **kwargs: Any
    ) -> CompletionResponseGen:
        pass


from LLM.SliME.llava.constants import IMAGE_PLACEHOLDER as IMAGE_PLACEHOLDER_SLIME
from LLM.SliME.llava.mm_utils import tokenizer_image_token as tokenizer_image_token_SLIME
class LLaVAHD(CustomLLM):
    context_window: int = 8192  # 上下文窗口大小
    num_output: int = 128  # 输出的token数量
    model_name: str = "LLaVAHD"  # 模型名称
    tokenizer: object = None  # 分词器
    model: object = None  # 模型
    image_processor: object = None # image_processor
    def __init__(self, pretrained_model_name_or_path=DEFAULT_LLM_MODEL):
        super().__init__()
        # GPU方式加载模型
        model_name = get_model_name_from_path_SLIME(pretrained_model_name_or_path)
        self.model_name = model_name
        tokenizer, model, image_processor, context_len = load_pretrained_model_LLaVAHD(pretrained_model_name_or_path, model_base=None, model_name=model_name)
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        # 得到LLM的元数据
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()  # 回调函数
    def complete(self, prompt: str, image: str, **kwargs: Any) -> CompletionResponse:
        pass

  #  @llm_chat_callback()  # 回调函数
    def chat(self, prompt, **kwargs: Any) -> ChatResponse:
        # 完成函数
        prompt, image = prompt[0], prompt[1]
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER_SLIME in prompt:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER_SLIME, image_token_se, prompt)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER_SLIME, DEFAULT_IMAGE_TOKEN, prompt)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + prompt
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt


        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates_SLIME[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if image is not None:
            # first message
            image = load_image(image)
            # image = image.resize((224, 224))
            image_tensor = process_images([image], self.image_processor, self.model.config)
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        else:
            image_tensor = None
        input_ids = tokenizer_image_token_SLIME(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).cuda()
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes = [image_tensor.shape] if image_tensor is not None else None,
                do_sample=False,
                temperature=0.2,
                max_new_tokens=512,
                num_beams=1,
                use_cache=False,
            )
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return outputs

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str,image: Optional[torch.FloatTensor] = None, **kwargs: Any
    ) -> CompletionResponseGen:
        pass

class OurLLM(CustomLLM):
    context_window: int = 8192  # 上下文窗口大小
    num_output: int = 128  # 输出的token数量
    model_name: str = "Mipha"  # 模型名称
    tokenizer: object = None  # 分词器
    model: object = None  # 模型
    image_processor: object = None # image_processor
    def __init__(self, pretrained_model_name_or_path=DEFAULT_LLM_MODEL):
        super().__init__()
        # GPU方式加载模型
        tokenizer, model, image_processor, context_len = load_pretrained_model(pretrained_model_name_or_path, model_base=None, model_name="Mipha-phi2")
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        # 得到LLM的元数据
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()  # 回调函数
    def complete(self, prompt: str, image: str, **kwargs: Any) -> CompletionResponse:
        pass

  #  @llm_chat_callback()  # 回调函数
    def chat(self, prompt, **kwargs: Any) -> ChatResponse:
        # 完成函数
        prompt, image = prompt[0], prompt[1]
        conv = conv_templates["phi"].copy()
        roles = conv.roles
        inp = f"{roles[0]}: {prompt}"

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        if image is not None:
            # first message
            image = load_image(image)
            # image = image.resize((224, 224))
            image_tensor = process_images([image], self.image_processor, self.model.config)
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
            image_tensor = None

        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).cuda()
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0,
                max_new_tokens=1024,
                streamer=streamer,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,  # End of sequence token
                pad_token_id=self.tokenizer.eos_token_id,  # Pad token
                stopping_criteria=[stopping_criteria]
            )
        outputs = self.tokenizer.decode(outputs[0, input_ids.shape[1]:]).strip()
        return outputs

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str,image: Optional[torch.FloatTensor] = None, **kwargs: Any
    ) -> CompletionResponseGen:
        pass
