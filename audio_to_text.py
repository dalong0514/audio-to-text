import os, time
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def audio_to_text():
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        output_dir='./output_dir')

    rec_result = inference_pipeline(audio_in='https://forgpt.oss-cn-hangzhou.aliyuncs.com/20231110%E5%85%AC%E5%8F%B8%E7%BB%8F%E8%90%A5%E4%BC%9A%E8%AE%AE-%E4%B8%87%E6%80%BB.mp3?OSSAccessKeyId=LTAI4GHKoH2Z5txaWP3NpNTx&Expires=1699985659&Signature=HYe99wKgoVHAA2JYOmsj4QwKVpE%3D')

if __name__ == '__main__':
    start_time = time.time()
    audio_to_text()
    end_time = time.time()
    print('OK!')
    print('Time Used: ' + str(end_time - start_time) + 's')