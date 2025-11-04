from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from PIL import Image
import os
import numpy as np
from llama_index.core import SimpleDirectoryReader
from build_rag import get_retriever_engine_from_local
app = FastAPI()

# 知识库路径
KNOWLEDGE_BASE_DIR = "storage/decompress"
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

# 计算像素差值
def calculate_pixel_difference(image1, image2):
    """
    Calculate the mean pixel difference between two images.
    """
    return np.mean(np.abs(np.array(image1) - np.array(image2)))

# 调整图片大小
def resize_image(image, size=(128, 128)):
    """
    Resize image to a uniform size for comparison.
    """
    return image.resize(size).convert('RGB')

# 比较图片与知识库的差值
def is_similar_to_knowledge_base(new_image, threshold=10):
    """
    Compare the new image with images in the knowledge base.

    :param new_image: PIL Image, the new image to compare.
    :param threshold: float, the pixel difference threshold.
    :return: True if similar, False if not similar.
    """
    for filename in os.listdir(KNOWLEDGE_BASE_DIR):
        kb_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        try:
            kb_image = Image.open(kb_path)
            kb_image_resized = resize_image(kb_image)
            new_image_resized = resize_image(new_image)
            diff = calculate_pixel_difference(new_image_resized, kb_image_resized)
            if diff < threshold:
                return True
        except Exception as e:
            print(f"Error processing knowledge base image {kb_path}: {e}")
    return False




# 保存图片到知识库
def save_to_knowledge_base(new_image, file_name, text):
    """
    Save the new image to the knowledge base.
    """
    save_img_path = os.path.join(KNOWLEDGE_BASE_DIR,"image", file_name)
    new_image.save(save_img_path)
    save_txt_path = os.path.join(KNOWLEDGE_BASE_DIR,"text", file_name.replace(".jpg", ".txt"))

    with open(save_txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    documents_img = SimpleDirectoryReader(input_files=[save_img_path]).load_data()
    for item in range(len(documents_img)):
        documents_img[item].metadata['file_name_img'] = save_img_path
        documents_img[item].metadata['file_name_text'] = save_txt_path

    documents_text = SimpleDirectoryReader(input_files=[save_txt_path]).load_data()
    for item in range(len(documents_text)):
        documents_text[item].metadata['file_name_img'] = save_img_path
        documents_text[item].metadata['file_name_text'] = save_txt_path
    num, content = get_retriever_engine_from_local()
    if num == 201:
        print(content)
    else:
        try:
            content["client"].index.refresh(
                documents_img,
                update_kwargs={"delete_kwargs": {"delete_from_docstore": True}},
            )
            content["client"].index.refresh(
                documents_text,
                update_kwargs={"delete_kwargs": {"delete_from_docstore": True}},
            )
            print("知识库更新成功")
        except Exception as e:
            print(e)


@app.post("/upload/")
async def upload_image(file: UploadFile, metadata: str = Form(...)):
    """
    Upload an image, compare it to the knowledge base, and add it if unique.
    """
    try:
        # 检查文件类型
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload JPEG or PNG.")
        # 增量更新索引

        # 打开上传的图片
        image = Image.open(file.file)

        # 检查是否与知识库中的图片相似
        threshold = 10
        is_similar = is_similar_to_knowledge_base(image, threshold)

        if is_similar:
            return JSONResponse(content={"message": "Image is similar to knowledge base, not added."}, status_code=200)

        # 如果不相似，将图片保存到知识库
        save_to_knowledge_base(image, file.filename, metadata)
        return JSONResponse(content={"message": "Image added to knowledge base."}, status_code=201)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Uvicorn 启动配置
if __name__ == "__main__":
    import uvicorn
    # 启动 uvicorn
    uvicorn.run(
        "update:app",  # 指定模块和 FastAPI 实例名称（文件名:实例名）
        host="0.0.0.0",  # 监听的主机地址
        port=8000,  # 监听的端口号
        reload=True  # 开启代码热重载（开发环境）
    )