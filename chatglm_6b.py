import os, time
from modelscope import AutoTokenizer, AutoModel, snapshot_download


def chatglm_text():
    model_dir = snapshot_download("ZhipuAI/chatglm3-6b", revision = "v1.0.0")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
    model = model.eval()
    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)
    response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
    print(response)


if __name__ == '__main__':
    start_time = time.time()
    chatglm_text()
    end_time = time.time()
    print('OK!')
    print('Time Used: ' + str(end_time - start_time) + 's')