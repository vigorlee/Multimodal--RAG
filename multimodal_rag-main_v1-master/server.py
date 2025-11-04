import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import shutil
import traceback

import gradio as gr
from gradio.components.chatbot import FileMessage
from gradio.data_classes import FileData

from build_rag import retrieve, get_retriever_engine, retrieve_image_to_image, get_retriever_engine_from_local, \
    retrieve_json
from PIL import Image
import numpy as np
import sys
from model_LLM import OurLLM, MiniCPM, MobileVLM, LLaVAHD

model_name = sys.argv[1]
if model_name == "MobileVLM":
    llm = MobileVLM()
elif model_name == "MiniCPM":
    llm = MiniCPM()
elif model_name == "LLaVA-HD":
    llm = LLaVAHD()

def complex_analysis(retriever_engine_dict, query_str, image, image_file_name):
    retriever_engine = retriever_engine_dict["retriever_engine"]
    json_response = retrieve_json(retriever_engine_dict["json_engine"],query_str)
    img = []
    txt = []
    if query_str != "":
        img, txt = retrieve(retriever_engine, query_str)
    if image is not None and img == []:
        # image 检索image只取image检索结果，忽略文本检索结果
        print("image 检索image只取image检索结果，忽略文本检索结果")
        image = Image.open(io.BytesIO(base64.b64decode(image))).convert('RGB')
        # 保存Image对象为文件
        image.save("storage/cache/%s" % (image_file_name))
        img1, txt1 = retrieve_image_to_image(retriever_engine, "storage/cache/%s"%(image_file_name))
        img = img1
        txt = txt1

    # 只取排序后每个子列表的第一个元素
    if txt != []:
        txt = sorted(txt, key=lambda x: x[1], reverse=True)
        print(txt)
        txt = [item[0] for item in txt][0]
        context_str = txt
        print("context_str", context_str)
    else:
        context_str = "没有上下文，直接回答"
    if img == []:
        return context_str, None, None
    img = sorted(img, key=lambda x: x[1], reverse=True)
    img = [item[0] for item in img]
    return context_str +" json:" + str(json_response), img, img[0]

def call_LLM(prompt, image_path, image=None, retriever_img_path=None):

    if image is not None and retriever_img_path is None:
        image = Image.open(io.BytesIO(base64.b64decode(image))).convert('RGB')
        image.save(image_path)

    if image is not None and retriever_img_path is not None:
        image1 = Image.open(io.BytesIO(base64.b64decode(image))).convert('RGB')
        image2 = Image.open(retriever_img_path)
        # Calculate the width and height of the new image
        new_width = image1.width + image2.width
        new_height = max(image1.height, image2.height)
        # Create a new image with the calculated dimensions
        new_image = Image.new('RGB', (new_width, new_height))
        # Paste images into the new image
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (image1.width, 0))
        # Save the new image
        new_image.save(image_path)

    if model_name == "MiniCPM" or model_name == "MiniCPM2":
        response = llm.chat({0: prompt, 1:image_path})[0]
        response = response.replace("<|endoftext|>","")
    else:
        response = llm.chat({0: prompt, 1:image_path}).replace("<|endoftext|>","")
    return response

import base64
import io
import os.path
import gradio as gr

from PIL import Image

import zipfile
def upload_img(file, _chatbot, _app_session):
    _app_session['sts'] = None
    _app_session['ctx'] = []
    with open(file, 'rb') as f1:
        base64_str = base64.b64encode(f1.read()) # str类型
    _app_session['file'] = base64_str
    _app_session['file_name'] = os.path.basename(file)
    return _chatbot, _app_session

def unzip(zip_file, output=None):
    """解压zip文件"""
    if not os.path.exists(output):
        os.mkdir(output)
    output = output or os.path.dirname(zip_file)  # 默认解压到当前目录同名文件夹中
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for zip_info in zip_ref.infolist():
            # 解决中文文件名乱码问题
            zip_info.filename = zip_info.filename.encode('cp437').decode('gbk')
            zip_ref.extract(zip_info, output)

def load_knowledge(_app_cfg, _chat_bot):
    if _app_cfg['ret'] is not None:
        _app_cfg['ret']['client'].close()
    code, message = get_retriever_engine_from_local()
    if code == 200:
        _app_cfg['ret'] = message
        _app_cfg['message'] = "success"
        _chat_bot.append(('', '知识库更新成功，现在你可以和我聊天啦！'))
        return "success", _app_cfg, _chat_bot
    else:
        _app_cfg['ret'] = None
        _app_cfg['message'] = message
        _chat_bot.append(('', '知识库更新失败，原因：%s' % message))
        return message, _app_cfg, _chat_bot

def respond(_app_cfg, _chat_bot, image_folder, text_folder):
    if _app_cfg['file'] is None:
        _app_cfg['ret'] = "您还没上传任何文件"
        return '', _app_cfg, _chat_bot

    debase = base64.b64decode(_app_cfg['file'])
    with open("storage/"+_app_cfg["file_name"], 'wb') as f1:
        f1.write(debase)  # str类型
    if _app_cfg["file_name"].endswith(".zip"):
        unzip("storage/"+_app_cfg["file_name"],"storage/decompress")

    if _app_cfg['ret'] is not None and not isinstance(_app_cfg['ret'], str):
        _app_cfg['ret']["client"].close()
    code, message = get_retriever_engine("storage/decompress", image_folder, text_folder)
    if code == 200:
        _app_cfg['ret'] = message
        _app_cfg['message'] = "success"
        _chat_bot.append(('', '知识库更新成功，现在你可以和我聊天啦！'))
        return "success", _app_cfg, _chat_bot
    else:
        _app_cfg['ret'] = None
        _app_cfg['message'] = message
        _chat_bot.append(('', '知识库更新失败，原因：%s'%message))
        return message, _app_cfg, _chat_bot

def respond1(message, _chat_bot, _app_cfg, prompt):

    if _app_cfg['ctx'] is None:
        _app_cfg['ctx'] = []

    base64_str = None
    image_file_name = None
    if message['files'] is not None:
        for item in message['files']:
            with open(item, 'rb') as f1:
                base64_str = base64.b64encode(f1.read()) # str类型
                image_file_name = os.path.basename(item)
    _question = message["text"]
    _context = _app_cfg['ctx'].copy()
    if _context:
        _context.append({"role": "user", "content": _question})
    else:
        _context = [{"role": "user", "content": _question}]

    print('<User>:', _question)
    if _app_cfg['ret'] is None:
        message_by_index, _app_cfg, _chat_bot = load_knowledge(_app_cfg, _chat_bot)

    if _app_cfg['ret'] is None:
        if image_file_name is None:
            _answer = call_LLM(_question,None)
            _app_cfg['ctx'] = _context
            _app_cfg['sts'] = 200
            _context.append({"role": "assistant", "content": _answer})
            _chat_bot.append((_question, _answer))

        else:
            _answer = call_LLM(_question,image_path=os.path.join("storage","cache", image_file_name),image=base64_str)
            _app_cfg['ctx'] = _context
            _app_cfg['sts'] = 200
            _app_cfg['img'] = os.path.join("storage", "cache", image_file_name)
            _context.append({"role": "assistant", "content": _answer, "img": _app_cfg['img']})
            _chat_bot.append(([_app_cfg['img'].__str__(), _question], _answer))
        print('<Assistant>:', _answer)

        #_chat_bot.append(("", [_app_cfg['img'].__str__(), _answer]))
        message["text"] = ""
        message['files'] = None
        return gr.MultimodalTextbox(interactive=True, file_types=["image"],
                                      placeholder="输入消息或者上传图片", show_label=False), _chat_bot, _app_cfg

    _knowledge, img, img_path = complex_analysis(_app_cfg['ret'], _question, base64_str, image_file_name)
    try:
        print("指令:",prompt + "  上下文是:"+_knowledge +"  问题是：" +_question )
        _answer = call_LLM("Prompt:"+prompt+". Context:"+_knowledge +". Question:" +_question, img_path)
    except Exception as e:
        traceback.print_exc()
        _answer = "调用大模型出现错误，错误原因: %s"%(e.__str__())
    _context.append({"role": "assistant", "content": _answer,"img": os.path.relpath(img_path) if img_path is not None else None})
    _app_cfg['ctx'] = _context
    _app_cfg['sts'] = 200
    _app_cfg['img'] = os.path.relpath(img_path) if img_path is not None else None
    print('<Assistant>:', _answer)
    message["text"] = ""
    message['files'] = None

    if img_path is not None:
        if image_file_name is not None:
            _chat_bot.append(([_app_cfg['img'].__str__(), ""], None))
        _chat_bot.append((_question, [os.path.relpath(img_path).__str__(),""]))
        _chat_bot.append((None, _answer))
    else:
        if image_file_name is not None:
            _chat_bot.append(([_app_cfg['img'].__str__(), ""], None))
        _chat_bot.append((_question, _answer))

    return gr.MultimodalTextbox(interactive=True, file_types=["image"],
                                      placeholder="输入消息或者上传图片", show_label=False), _chat_bot, _app_cfg

def regenerate_button_clicked( _chat_bot, _app_cfg, image_folder, text_folder):
    message, app_session,_chat_bot = respond(_app_cfg, _chat_bot, image_folder, text_folder)
    return _chat_bot, app_session

def delete_know_base(_app_cfg, _chat_bot):
    if _app_cfg['ret'] is not None:
        _app_cfg['ret']["client"].close()
    shutil.rmtree("storage/decompress")
    shutil.rmtree("qdrant_mm_db/collection")
    _chat_bot.append(('', '知识库删除成功。'))
    return _app_cfg, _chat_bot

def create_component(params):
    return gr.Button(
        value=params['value'],
        interactive=True
    )

with gr.Blocks() as funclip_service:
    with gr.Column():
        with gr.Row():
            bt_file = gr.File(label="上传知识库 .zip 格式文件")
            konwbase = create_component({'value': '向量化'})
            rmkonwbase = create_component({'value': '删除知识库'})

        with gr.Column():
            image_folder = gr.Textbox(label="image_folder",
                                value="image",
                                interactive=True)

            text_folder = gr.Textbox(label="text_folder",
                                value="text",
                                interactive=True)

        with gr.Column():
            prompt = gr.Textbox(label="prompt",
                                value="""You're a navigation assistant robot. Based on the information and questions, respond with coordinates if the text includes object coordinates (x, y). And a json, "name" is room name,"top_left" and "bottom_right" is the room range, "aps" is access point, "object" listed object in room.""",
                                interactive=True)
            chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"],
                                      placeholder="输入消息或者上传图片", show_label=False)

    with gr.Column():
        with gr.Row():
            app_session = gr.State({'sts': None, 'ctx': None, 'img': None,'ret':None,'file':None})
            chat_bot = gr.Chatbot(label=f"Chat with {model_name}")

    konwbase.click(
        regenerate_button_clicked,
       [chat_bot, app_session, image_folder, text_folder],
        [chat_bot, app_session]
    )
    rmkonwbase.click(
        delete_know_base,
        [app_session, chat_bot],
        [app_session, chat_bot]
    )
    chat_input.submit(
        respond1,
        [chat_input, chat_bot, app_session, prompt],
        [chat_input, chat_bot, app_session]
    )
    bt_file.upload(lambda: None, None, chat_bot, queue=False).then(upload_img,
                                                                  inputs=[bt_file, chat_bot, app_session],
                                                                  outputs=[chat_bot, app_session])

if __name__ == '__main__':
    funclip_service.launch(share=False)
