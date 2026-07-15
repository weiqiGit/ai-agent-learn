import os
from dotenv import load_dotenv
from app.core.rag_engine import (
    load_document,
    split_texts,
    get_vector_store,
    create_qa_chain,
)


load_dotenv()
# 1. 创建测试文档
test_content = """
# 公司年假制度

 1. 年假申请条件：入职满一年后可申请年假。
 2. 年假天数：工龄1-5年每年5天，5-10年每年10天，10年以上每年15天。
 3. 申请流程：提前3天在OA系统提交申请，经部门主管审批后生效。
 4. 未休年假：当年未休完的年假可顺延至次年3月31日，逾期作废。
 """

os.makedirs("./uploads", exist_ok=True)
with open("./uploads/test_doc.txt", "w", encoding="utf-8") as f:
    f.write(test_content)

print("1. 加载文档...")
documents = load_document("./uploads/test_doc.txt")
print(f"   加载了 {len(documents)} 页")

print("2. 切片...")
chunks = split_texts(documents)
print(f"   切成了 {len(chunks)} 块")

print("3. 向量化并存储...")
vectordb = get_vector_store(chunks)
print("   向量库新建或查询成功")

print("4. 创建问答链...")
qa_chain = create_qa_chain(vectordb)

print("5. 开始提问...")
question = "年假怎么申请？"
result = qa_chain({"query": question})
print(f"\n问题：{question}")
print(f"回答：{result['result']}")
print(f"引用来源：{len(result['source_documents'])} 个文档块")
