import base64
import os
import traceback
from build_rag import update_nodes
from server import call_LLM, load_knowledge, complex_analysis
import io
from PIL import Image




"""
message : json
"""
# Recived query message Q from users;
def query(message):
    global retriever_engine_dict
    """
        message: json
            question: str
        return: Position
    """
    question = message['question']
    # Query Q in the knowledge base {K} for the results(I, T , P , t );
    _knowledge, img, img_path = complex_analysis(retriever_engine_dict, question, None, None)
    # Send history Position P to Robot;
    return _knowledge

"""
message : json
"""
# When Robit arrived, recived image I(i) and Position P(i);
def arrived_update_and_generate(message):
    global retriever_engine_dict
    """
        message: json
            image: base64, Position: str, object: str, frame/timestamp: str
        return: Yes or No
    """
    base64_str = message['image']
    position = message['position']


    object = message['object']
    timestamp = message['timestamp']
    prompt = f"Left image is the newest image, right is the old image, is the two sense has the same {object}? Answer yes or no"
    _question = ""
    _knowledge, img, img_path = complex_analysis(retriever_engine_dict, _question, base64_str, timestamp+".jpg")
    if img is not None:
        try:
            print("上下文是:"+_knowledge +"。问题是：" +_question + prompt)
            _answer  = call_LLM("Context:"+_knowledge +". Question:" +_question + prompt, image=base64_str, image_path = os.path.join("storage","cache", "outcome.jpg"), retriever_img_path = img_path)
        except Exception as e:
            traceback.print_exc()
            _answer = "调用大模型出现错误，错误原因: %s"%(e.__str__())
    else:
        # 没找到直接插入
        _answer = "yes"

    if "yes" in _answer.lower():
        image = Image.open(io.BytesIO(base64.b64decode(base64_str))).convert('RGB')
        file_path = f"storage/knowledgebase/{timestamp}.jpg"
        image.save(file_path)
        # Get T(i)  describe the sense I by VLM
        _answer = call_LLM("Describe the indoor sense.", image=base64_str,image_path=os.path.join("storage", "cache", "outcome.jpg"), retriever_img_path=img_path)
        # Insert nodes (I(i),T(i) , P(i), t(i)) to update knowledge base {K}
        update_nodes(file_path, _answer+f". The coordinate is {position}", retriever_engine_dict)
        # Send to Robot successed to find
        return True

    elif "no" in _answer.lower():
        image = Image.open(io.BytesIO(base64.b64decode(base64_str))).convert('RGB')
        file_path = f"storage/knowledgebase/{timestamp}.jpg"
        image.save(file_path)
        # Get T(i)  describe the sense I by VLM
        _answer = call_LLM("Describe the indoor sense.", image=base64_str, image_path=os.path.join("storage", "cache", "outcome.jpg"), retriever_img_path=img_path)
        # Insert nodes (I(i),T(i) , P(i), t(i)) to update knowledge base {K}
        update_nodes(file_path, _answer + f". The coordinate is {position}", retriever_engine_dict)
        # Send to Robot Failed to find
        return False