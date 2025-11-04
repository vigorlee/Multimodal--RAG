import json
import os
import traceback

from PIL import Image
from llama_index.core import Settings
from llama_index.core import ServiceContext, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.indices.struct_store import JSONQueryEngine
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.readers.json import JSONReader

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from multi_modal_lndex.base import MultiModalVectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
# Create a local Qdrant vector store
import qdrant_client



def retrieve(retriever_engine, query_str):
    retrieval_results = retriever_engine.retrieve(query_str)
    retrieved_image = []
    retrieved_text = []

    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            #print(res_node.score,":",res_node.text)
            if res_node.score > 0.05:
                retrieved_image.append([res_node.node.metadata["file_path"],res_node.score])
                if "file_name_text" in res_node.node.metadata.keys():
                    with open(res_node.node.metadata["file_name_text"], "r", encoding="utf-8") as f:
                        retrieved_text.append([f.read(),res_node.score])
        else:
            #print(res_node.score,":",res_node.text)
            pass
            #if res_node.score > 0.5:
            #    retrieved_text.append([res_node.text,res_node.score])
            #    if "file_name_img" in res_node.node.metadata.keys():
            #        retrieved_image.append([res_node.node.metadata["file_name_img"],res_node.score])
        #print(res_node.score)

    return retrieved_image, retrieved_text

def retrieve_json(retriever_engine, query_str):
    retrieval_results = retriever_engine.retrieve(query_str)
    retrieved_text = []

    for res_node in retrieval_results:
        if res_node.score > 0.09:
            retrieved_text.append(res_node.text)
            #if res_node.score > 0.5:
            #    retrieved_text.append([res_node.text,res_node.score])
            #    if "file_name_img" in res_node.node.metadata.keys():
            #        retrieved_image.append([res_node.node.metadata["file_name_img"],res_node.score])
        #print(res_node.score)

    return retrieved_text

def retrieve_image_to_image(retriever_engine, image):
    retrieval_results = retriever_engine.image_to_image_retrieve(image)
    retrieved_image = []
    retrieved_text = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            if res_node.score > 0.05:
                retrieved_image.append([res_node.node.metadata["file_path"],res_node.score])
                if "file_name_text" in res_node.node.metadata.keys():
                        with open(res_node.node.metadata["file_name_text"], "r", encoding="utf-8") as f:
                            retrieved_text.append([f.read(),res_node.score])
        else:
            if res_node.score > 0.5:
                retrieved_text.append(res_node.text)
                if "file_name_img" in res_node.node.metadata.keys():
                    retrieved_image.append([res_node.node.metadata["file_name_img"], res_node.score])

    return retrieved_image, retrieved_text

def get_all_files(path):
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance


class Client():
    _instance = None
    qdrant_client = None
    def __init__(self):
        if self.qdrant_client is None:
            self.qdrant_client = qdrant_client.QdrantClient(host="localhost", port=6333)
        else:
            self.close()
            self.qdrant_client = qdrant_client.QdrantClient(host="localhost", port=6333)
        self.index = None
        self.json_index = None
    def close(self):
        if self.qdrant_client is not None:
            self.qdrant_client.close()
            self.qdrant_client = None


    def __new__(cls):
        if not cls._instance:
            cls._instance = super(Client, cls).__new__(cls)
            # 初始化操作
        return cls._instance
from llama_index.core.base.embeddings.base import BaseEmbedding
def get_retriever_engine_from_local():
    try:
        client = Client()
        Settings.llm = None
        Settings.embed_model = None
        text_store = QdrantVectorStore(
            client=client.qdrant_client, collection_name="text_collection"
        )
        image_store = QdrantVectorStore(
            client=client.qdrant_client, collection_name="image_collection"
        )
        json_store = QdrantVectorStore(
            client=client.qdrant_client, collection_name="json_collection"
        )
        storage_context_json = StorageContext.from_defaults(vector_store=json_store
                                                            )
        service_context = ServiceContext.from_defaults(
            embed_model=HuggingFaceEmbedding(r"E:\model\bge-small-zh-v1.5"), llm=None
        )
        index_json = VectorStoreIndex.from_vector_store(
            vector_store=json_store, storage_context=storage_context_json, service_context=service_context
        )

        storage_context = StorageContext.from_defaults(
            vector_store=text_store, image_store=image_store
        )
        index = MultiModalVectorStoreIndex.from_vector_store(
            image_embed_model="siglip",
            storage_context=storage_context, service_context=None
        )
        client.index = index
        client.json_index = index_json

        retriever_engine = index.as_retriever(
            similarity_top_k=3, image_similarity_top_k=3
        )
        retriever_engine_json = index_json.as_retriever(
            similarity_top_k=1
        )
    except Exception as e:
        traceback.print_exc()
        return 201, e
    return 200, {"retriever_engine":retriever_engine, "client": client, "index":index,"json_engine":retriever_engine_json }


def update_nodes(file_path, _answer, cli_json):
    file_list = [file_path]
    client = cli_json["client"]
    index = cli_json["index"]
    try:
        documents = SimpleDirectoryReader(input_files=file_list).load_data()
        for item in range(len(documents)):
            documents[item].metadata['file_name_img'] = file_path[0]
            with open(file_path[0].replace(".jpg",".txt"), "r", encoding="utf-8") as f:
                f.write(_answer)
            documents[item].metadata['file_name_text'] = file_path[0].replace(".jpg",".txt")
        parser = SentenceSplitter()
        nodes = parser.get_nodes_from_documents(documents)
        index.insert_nodes(nodes=nodes)
        retriever_engine = index.as_retriever(
            similarity_top_k=3, image_similarity_top_k=3
        )
    except Exception as e:
        client.close()
        return 201, e
    return 200, {"retriever_engine":retriever_engine, "client": client, "index":index}



def get_path_except_last_two(path, separator='/'):
    path_elements = path.split(separator)
    # 去掉空字符串元素
    path_elements = [element for element in path_elements if element]
    # 如果路径中有至少两个元素，则返回除了最后两个元素外的所有元素
    if len(path_elements) >= 2:
        # 拼接除最后两个元素外的所有元素
        return separator.join(path_elements[:-2]) + separator
    else:
        return ''

def get_retriever_engine(path, image_folder, text_folder):
    global client
    file_list = get_all_files(path)
    try:

        client = Client() #qdrant_client.QdrantClient(path="qdrant_mm_db")

        Settings.llm = None
        Settings.embed_model = None
        documents = SimpleDirectoryReader(input_files=file_list).load_data()

        if image_folder is not None and text_folder is not None:
            for item in range(len(documents)):
                dir_file_img = os.listdir(f"storage/decompress/{image_folder}")
                dir_file_text = os.listdir(f"storage/decompress/{text_folder}")
                img_lastfix = dir_file_img[-1].split(".")[-1]
                text_lastfix = dir_file_text[-1].split(".")[-1]
                print("documents[item].metadata['file_name_img']  :","storage/decompress/" + f"{image_folder}" + "/" + os.path.basename(file_list[item]).replace(text_lastfix,img_lastfix))
                documents[item].metadata['file_name_img'] ="storage/decompress/" + f"{image_folder}" + "/" + os.path.basename(file_list[item]).replace(text_lastfix,img_lastfix)
                #file_list[item].replace(text_folder, image_folder).replace(text_lastfix,img_lastfix)
                print("documents[item].metadata['file_name_text']   :", "storage/decompress/" + f"{text_folder}" + "/" + os.path.basename(file_list[item]).replace(img_lastfix,text_lastfix))
                documents[item].metadata['file_name_text'] = "storage/decompress/" + f"{text_folder}" + "/" + os.path.basename(file_list[item]).replace(img_lastfix,text_lastfix)
        parser = SentenceSplitter()
        nodes = parser.get_nodes_from_documents(documents)

        text_store = QdrantVectorStore(
            client=client.qdrant_client, collection_name="text_collection" # 设置向量长度
        )
        image_store = QdrantVectorStore(
            client=client.qdrant_client, collection_name="image_collection"
        )
        json_store = QdrantVectorStore(
            client=client.qdrant_client, collection_name="json_collection"
        )

        storage_context = StorageContext.from_defaults(
            vector_store=text_store, image_store=image_store
        )
        storage_context_json = StorageContext.from_defaults(vector_store=json_store
        )
        index = MultiModalVectorStoreIndex(
            nodes=nodes,image_embed_model="siglip",
            storage_context=storage_context,service_context=None
        )

        retriever_engine = index.as_retriever(
            similarity_top_k=3, image_similarity_top_k=3
        )
        reader = JSONReader()
        documents_ = []
        for item in os.listdir("storage/regions_output"):
            documents = reader.load_data(input_file=os.path.join("storage/regions_output",item), extra_info={})
            documents_.extend(documents)
        # create the pipeline with transformations
        pipeline = IngestionPipeline(
            transformations=[
                HuggingFaceEmbedding(r"E:\model\bge-small-zh-v1.5"),
            ]
        )

        # run the pipeline
        nodes = pipeline.run(documents=documents_)
        service_context = ServiceContext.from_defaults(
            embed_model=HuggingFaceEmbedding(r"E:\model\bge-small-zh-v1.5"),llm=None
        )
        index_json = VectorStoreIndex(
            nodes=nodes, storage_context=storage_context_json, service_context=service_context
        )
        client.index = index
        client.json_index = index_json

        retriever_engine_json = index_json.as_retriever(
            similarity_top_k=1
        )
    except Exception as e:
        print(e)
        traceback.print_exc()
        client.close()
        return 201, e
    return 200, {"retriever_engine":retriever_engine, "client": client,"index":index,"json_engine":retriever_engine_json}



