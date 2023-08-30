import os, time
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def audio_to_text():
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        output_dir='./output_dir')

    rec_result = inference_pipeline(audio_in='https://forgpt.oss-cn-hangzhou.aliyuncs.com/20230830%E7%94%B2%E7%B1%BB%E4%BB%93%E5%BA%93%E6%A8%A1%E5%9D%97%E6%96%B9%E6%A1%88%E8%AF%84%E5%AE%A1.wav?OSSAccessKeyId=LTAI4GHKoH2Z5txaWP3NpNTx&Expires=1693374361&Signature=K9tDj02NdC5PSuc4NHd6QEdthzQ%3D')

if __name__ == '__main__':
    start_time = time.time()
    audio_to_text()
    end_time = time.time()
    print('OK!')
    print('Time Used: ' + str(end_time - start_time) + 's')