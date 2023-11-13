import os, time
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def audio_to_text():
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        output_dir='./output_dir')

    rec_result = inference_pipeline(audio_in='https://forgpt.oss-cn-hangzhou.aliyuncs.com/20230731M2-Ultra-AI%E5%A4%A7%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%96%B0%E5%B8%8C%E6%9C%9B.mp3?OSSAccessKeyId=LTAI4GHKoH2Z5txaWP3NpNTx&Expires=1699892994&Signature=cxjzvFS7TmpMB7Gk86yYkqC4w3s%3D')

if __name__ == '__main__':
    start_time = time.time()
    audio_to_text()
    end_time = time.time()
    print('OK!')
    print('Time Used: ' + str(end_time - start_time) + 's')