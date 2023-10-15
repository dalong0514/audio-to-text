import os, time
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def audio_to_text():
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        output_dir='./output_dir')

    rec_result = inference_pipeline(audio_in='https://forgpt.oss-cn-hangzhou.aliyuncs.com/20231015%E9%98%B3%E5%BF%97%E5%B9%B3AI%E9%97%AD%E9%97%A8%E5%88%86%E4%BA%AB%E4%BC%9A01.mp3?OSSAccessKeyId=LTAI4GHKoH2Z5txaWP3NpNTx&Expires=1697368453&Signature=bSiaJb8wsMTOn38cINE2E4nL6ds%3D')

if __name__ == '__main__':
    start_time = time.time()
    audio_to_text()
    end_time = time.time()
    print('OK!')
    print('Time Used: ' + str(end_time - start_time) + 's')