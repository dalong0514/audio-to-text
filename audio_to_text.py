import os, time
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def audio_to_text():
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        output_dir='./output_dir')

    rec_result = inference_pipeline(audio_in='https://forgpt.oss-cn-hangzhou.aliyuncs.com/20231110%E4%B8%8E%E6%9D%8E%E5%A5%95%E9%93%AE%E4%BA%A4%E6%B5%81%E5%BB%BA%E7%AD%91%E7%94%B2%E7%B1%BB%E4%BB%93%E5%BA%93%E9%9C%80%E6%B1%82.mp3?OSSAccessKeyId=LTAI4GHKoH2Z5txaWP3NpNTx&Expires=1699870936&Signature=3WAI%2B3qSKCPMc2y5HxuSgX89Cb0%3D')

if __name__ == '__main__':
    start_time = time.time()
    audio_to_text()
    end_time = time.time()
    print('OK!')
    print('Time Used: ' + str(end_time - start_time) + 's')