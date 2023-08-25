from modelscope.pipelines import pipeline
word_segmentation = pipeline('word-segmentation')
input_str = '今天天气不错，适合出去游玩'
print(word_segmentation(input_str))
