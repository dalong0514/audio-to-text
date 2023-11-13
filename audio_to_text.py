import os, time
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def audio_to_text():
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        output_dir='./output_dir')

    rec_result = inference_pipeline(audio_in='https://forgpt.oss-cn-hangzhou.aliyuncs.com/20230803%E4%B8%8E%E5%90%B4%E6%80%BB%E6%B1%87%E6%8A%A5%E5%90%84%E4%B8%93%E4%B8%9A%E4%BD%9C%E4%B8%9A%E7%BA%BF%E6%A1%86%E6%9E%B6.mp3?OSSAccessKeyId=LTAI4GHKoH2Z5txaWP3NpNTx&Expires=1699898494&Signature=f5o0QK7oxeRjNBCA%2BASJpdXkj3I%3D')

if __name__ == '__main__':
    start_time = time.time()
    audio_to_text()
    end_time = time.time()
    print('OK!')
    print('Time Used: ' + str(end_time - start_time) + 's')