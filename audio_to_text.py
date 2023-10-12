import os, time
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def audio_to_text():
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        output_dir='./output_dir')

    rec_result = inference_pipeline(audio_in='https://forgpt.oss-cn-hangzhou.aliyuncs.com/20190712%E5%BC%A0%E6%B6%9B%E7%9F%B3%E5%8C%96%E8%A7%84%E5%AE%A3%E8%B4%AF1.mp3?OSSAccessKeyId=LTAI4GHKoH2Z5txaWP3NpNTx&Expires=1697126295&Signature=l583ShCS%2B%2BCWscSAL5%2B6S%2BjjnqI%3D')

if __name__ == '__main__':
    start_time = time.time()
    audio_to_text()
    end_time = time.time()
    print('OK!')
    print('Time Used: ' + str(end_time - start_time) + 's')