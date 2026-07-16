# RAG 核心逻辑（加载、切片、向量库）
import os
import json

# 读取 PDF 和 TXT 文件，转成 Document 对象
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# 把长文档切成小块
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 向量数据库，存向量，查相似
from langchain_community.vectorstores import Chroma

# 用 DeepSeek 生成回答
from langchain_community.chat_models import ChatOpenAI

# 检索链
from langchain.chains import RetrievalQA

# 把文本转化成向量
from langchain_community.embeddings import ZhipuAIEmbeddings
from typing import List, AsyncIterator
from dotenv import load_dotenv


# 向量库持久化目录
PERSIST_DIR = "./chroma_db"
DEFAULT_SESSION = "default"

load_dotenv()
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量 ZHIPUAI_API_KEY")
embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)


# 根据文件格式，选择不同的加载器，把文件读成Document对象
def load_document(file_path: str):
    if file_path.endswith(".pdf"):
        documents = PyPDFLoader(file_path).load()
    elif file_path.endswith(".txt"):
        documents = TextLoader(file_path, encoding="utf-8").load()
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")
    # 为所有 Document 添加自定义 metadata
    for doc in documents:
        doc.metadata["uploaded_by"] = DEFAULT_SESSION
    return documents


# 把文件切成小块，每块500字符，相邻两块重叠50个字符
def split_texts(documents, chunk_size=500, chunk_overlap=50):
    # 创建一个切片器实例
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    # 给每个块加 chunk_id
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


_vectordb_cache = None


# 获取会话级向量库+添加数据
def get_vector_store(chunks=None):
    global _vectordb_cache
    if _vectordb_cache is not None:
        # 如果有新文档要添加，追加到缓存
        if chunks:
            _vectordb_cache.add_documents(chunks)
        return _vectordb_cache
    # 获取全局向量库（不存在则自动创建）
    # Chroma 会自动创建目录和空库
    _vectordb_cache = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    # 有新文档就追加
    if chunks:
        _vectordb_cache.add_documents(chunks)
    return _vectordb_cache


# 获取知识库所有已上传的文件及块数
def get_files_list() -> List[dict]:
    try:
        vectordb = get_vector_store()
        # 包含ids,documents,metadatas
        all_docs = vectordb.get()
        # 获取所有文档的 metadata，包含source
        metadatas = all_docs.get("metadatas", [])

        # 按 source 分组统计
        file_map = {}
        for meta in metadatas:
            source = meta.get("source", "")
            if source:
                # 只取文件名，不要路径
                file_name = os.path.basename(source)
                if file_name not in file_map:
                    file_map[file_name] = {
                        "name": file_name,
                        "source": source,
                        "chunks": 0,
                    }
                file_map[file_name]["chunks"] += 1

        return list(file_map.values())
    except Exception as e:
        print(f"获取文件列表失败: {e}")
        return []


# 删除指定文件的所有块
def delete_file_from_store(file_name: str) -> int:
    try:
        vectordb = get_vector_store()
        # 获取该文件的所有 chunk IDs
        # 注意：Chroma 的 where 条件匹配 metadata 中的 source 字段
        # 但 source 存的是完整路径，所以需要构造 where 条件
        # 方法：先获取所有文档，再过滤
        all_docs = vectordb.get()
        ids = all_docs.get("ids", [])
        metadatas = all_docs.get("metadatas", [])

        # 找出匹配的文件名对应的 ids
        ids_to_delete = []
        # 配对遍历
        for doc_id, meta in zip(ids, metadatas):
            source = meta.get("source", "")
            if source and os.path.basename(source) == file_name:
                ids_to_delete.append(doc_id)

        if not ids_to_delete:
            return 0

        # 删除这些块
        vectordb._collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)
    except Exception as e:
        print(f"删除文件 {file_name} 失败: {e}")
        return 0


# 流式问答
async def stream_chat(question: str, context: str) -> AsyncIterator[str]:
    try:
        if context:
            prompt = f"""你是一个专业的知识库助手。请根据以下文档内容回答用户的问题。
            如果文档中没有相关信息，请如实说"文档中未提及"。
            不要编造答案，不要使用文档外的知识。

            文档内容：
            {context}

            用户问题：{question}"""
        else:
            prompt = question
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=0.3 if context else 0.7,
            streaming=True,
        )
        async for chunk in llm.astream(prompt):
            content = chunk.content
            if content:
                yield f"data: {json.dumps({'content': content})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# rag流式
async def stream_rag_answer(question: str) -> AsyncIterator[str]:
    try:
        vectordb = get_vector_store()
        # 把向量库包装成 LangChain 检索器，需要返回4个最相似的文档块
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        # 组件运行，docs的类型是List[Document]，有page_content和page_meta
        docs = retriever.invoke(question)

        # 提取来源：——遍历docs，取出source中的文件名，并用set转成集合去重，再用list转成列表
        sources = list(
            set(
                [
                    os.path.basename(doc.metadata.get("source", "未知来源"))
                    for doc in docs
                ]
            )
        )
        context = "\n\n".join([doc.page_content for doc in docs])
        # 异步流式调用 LLM，每次返回一个字符块,异步迭代，每收到一个字符块就执行一次循环
        async for chunk in stream_chat(question, context):
            yield chunk
        # 返回来源+done
        yield f"data: {json.dumps({'sources': sources, 'done': True})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# 普通流式
async def stream_answer(question: str) -> AsyncIterator[str]:
    try:
        async for chunk in stream_chat(question, ""):
            yield chunk
        yield f"data: {json.dumps({'done': True})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# 已弃用  非流式——创建问答链，自动检索，拼接prompt，调用llm，返回answer
def create_qa_chain(vectordb):
    # 初始化聊天模型
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        # 0保守，1放飞
        temperature=0.3,
    )
    # 创建检索器，表示每次检索返回最相似的 3 个文档块
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    # 用 LangChain 创建一个 RAG 问答链
    return RetrievalQA.from_chain_type(
        llm=llm,
        # staff是是完全拼接模式
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
