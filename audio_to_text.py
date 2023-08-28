import os, time
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def audio_to_text():
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        output_dir='./output_dir')

    rec_result = inference_pipeline(audio_in='https://forgpt.oss-cn-hangzhou.aliyuncs.com/20230625%E4%B8%8E%E7%8E%8B%E5%BB%BA%E6%85%A7%E4%BA%A4%E6%B5%81%E7%94%B2%E7%B1%BB%E4%BB%93%E5%BA%93.mp3?OSSAccessKeyId=LTAI4GHKoH2Z5txaWP3NpNTx&Expires=1693232671&Signature=Y7hyTzNT9A67zYDOxMg2wU%2Bob3M%3D')

if __name__ == '__main__':
    start_time = time.time()
    audio_to_text()
    end_time = time.time()
    print('OK!')
    print('Time Used: ' + str(end_time - start_time) + 's')