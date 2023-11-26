import os, time
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def audio_to_text():
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        output_dir='./output_dir')

    rec_result = inference_pipeline(audio_in='https://forgpt.oss-cn-hangzhou.aliyuncs.com/20230705%E8%88%92%E4%BC%9F%E6%9D%B0%E5%9F%B9%E8%AE%ADHAZOP.m4a?OSSAccessKeyId=LTAI4GHKoH2Z5txaWP3NpNTx&Expires=1701023897&Signature=1VvihORJaA4wk%2FUmw9FQ83N9WBM%3D')

if __name__ == '__main__':
    start_time = time.time()
    audio_to_text()
    end_time = time.time()
    print('OK!')
    print('Time Used: ' + str(end_time - start_time) + 's')