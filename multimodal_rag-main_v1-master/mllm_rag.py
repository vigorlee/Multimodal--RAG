import torch
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback
from llama_index.core.node_parser import SentenceSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, TextStreamer
from typing import Optional, List, Mapping, Any

from mipha.constants import IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN
from mipha.conversation import conv_templates
from mipha.eval.model_qa import KeywordsStoppingCriteria
from mipha.mm_utils import tokenizer_image_token, process_images
from mipha.serve.cli import load_image
from multi_modal_lndex.base import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext, Settings, ServiceContext
import qdrant_client
from mipha.model.builder import load_pretrained_model
from llama_index.core.llms import CustomLLM,CompletionResponse,CompletionResponseGen,LLMMetadata

# set context window size
context_window = 2048
# set number of output tokens
num_output = 256
model_name = "Mipha"

#tokenizer = AutoTokenizer.from_pretrained(f"D:\zzk\LLM\Mipha")
# model = AutoModel.from_pretrained("Qwen-7B-Chat", trust_remote_code=True, device='cuda')
#model = AutoModelForCausalLM.from_pretrained(f"D:\zzk\LLM\Mipha", device_map="auto",  bf16=True).eval()

class OurLLM(CustomLLM):
    context_window: int = 8192  # 上下文窗口大小
    num_output: int = 128  # 输出的token数量
    model_name: str = "Mipha"  # 模型名称
    tokenizer: object = None  # 分词器
    model: object = None  # 模型
    image_processor: object = None # image_processor
    def __init__(self, pretrained_model_name_or_path=f"D:\Mipha\Mipha-phi2"):
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
        # 完成函数
        print("完成函数")

        conv = conv_templates["phi"].copy()
        roles = conv.roles
        inp = f"{roles[0]}: {prompt}"

        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        if image is not None:
            # first message
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
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

        return CompletionResponse(text=outputs)


  #  @llm_chat_callback()  # 回调函数
    def chat(self, prompt, **kwargs: Any) -> ChatResponse:
        # 完成函数
        print("完成函数")
        prompt, image = prompt[0], prompt[1]
        conv = conv_templates["phi"].copy()
        roles = conv.roles
        inp = f"{roles[0]}: {prompt}"
        #prompt = conv.get_prompt()
        image = load_image(image)
        #image = image.resize((224, 224))

        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        if image is not None:
            # first message
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
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
        # 流式完成函数
        print("流式完成函数")
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').cuda()  # GPU方式
        outputs = self.model(input_ids=inputs,image=image)
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').cuda()  # GPU方式
        # inputs = self.tokenizer.encode(prompt, return_tensors='pt')  # CPU方式
        outputs = self.model.generate(inputs, max_length=self.num_output)
        print()
      #  response = self.tokenizer.decode(outputs[0])
        for token in response:
            yield CompletionResponse(text=token, delta=token)



# Create a local Qdrant vector store
client = qdrant_client.QdrantClient(path="qdrant_mm_db")

# if you only need image_store for image retrieval,
# you can remove text_sotre
text_store = QdrantVectorStore(
    client=client, collection_name="text_collection"
)
image_store = QdrantVectorStore(
    client=client, collection_name="image_collection"
)

storage_context = StorageContext.from_defaults(
    vector_store=text_store, image_store=image_store
)

Settings.llm = None
Settings.embed_model=None
# Load text and image documents from local folder
documents = SimpleDirectoryReader("./images").load_data()
# Create the MultiModal index
# parse nodes
#
parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)
llm = OurLLM()
print(llm.chat({0: "image say what", 1: "images/img_2.png"}))
service_context = ServiceContext.from_defaults(llm=llm,embed_model=None)

index = MultiModalVectorStoreIndex(
    nodes=nodes,image_embed_model="siglip",embed_model=None,
    storage_context=storage_context,service_context=service_context
)

retriever_engine = index.as_retriever(
    similarity_top_k=3, image_similarity_top_k=3
)

# retrieve more information from the GPT4V response

# if you only need image retrieval without text retrieval
# you can use `text_to_image_retrieve`
# retrieval_results = retriever_engine.text_to_image_retrieve(response)

from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode
import matplotlib.pyplot as plt
import os
from PIL import Image

def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)
            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])
            plt.savefig("image.png")
            images_shown += 1
            if images_shown >= 7:
                break
def retrieve(retriever_engine, query_str):
    retrieval_results = retriever_engine.retrieve(query_str)
    retrieved_image = []
    retrieved_text = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
        else:
            display_source_node(res_node, source_length=200)
            retrieved_text.append(res_node.text)

    return retrieved_image, retrieved_text

query_str = "a photo of two cat"
img, txt = retrieve(retriever_engine=retriever_engine, query_str=query_str)

context_str = "".join(txt)
plot_images(img)

